# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config

from ...common import DummyPipelineDataset, PipelineDataset, benchmark_model


class T5_encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(self, input_ids, attention_mask):
        return self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)


class T5_decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.t5 = model

    def forward(
        self,
        decoder_input_ids,
        decoder_attention_mask,
        encoder_last_hidden_state,
        decoder_encoder_attention_mask,
        *past_key_values,
    ):
        presents = []
        pkv = []
        for i, _ in enumerate(self.t5.decoder.block):
            pkv.append(tuple([past_key_values[(i * 4) + j] for j in range(4)]))

        outputs = self.t5.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=decoder_encoder_attention_mask,
            past_key_values=pkv,
        )
        sequence_output = outputs[0]
        presents = outputs[1]
        if self.t5.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5.model_dim**-0.5)

        lm_logits = self.t5.lm_head(sequence_output)
        return lm_logits, *presents


@benchmark_model(configs=["small", "base", "large"])
def flant5_past_cache_enc_dec(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda.pybudaglobal import TILE_DIM

        # Add PyBUDA configurations
        os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
        os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
        os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "120000"
        os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
        os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "26000"
        os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        os.environ["TT_BACKEND_PROFILER"] = "1"
        os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
        os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_t_streaming = True
        compiler_cfg.enable_tvm_cpu_fallback = False
        compiler_cfg.default_df_override = pybuda._C.Float16_b
        compiler_cfg.default_dram_parameters = False
        compiler_cfg.input_queues_on_host = True
        compiler_cfg.enable_amp_light()
        compiler_cfg.compile_subgraphs = True
        compiler_cfg.enable_link_past_cache_ios = True

    # Set model parameters based on chosen task and model configuration
    if task in ["na", "text_classification", "text_summarization"]:
        if config == "small":
            model_name = "google/flan-t5-small"
        elif config == "base":
            model_name = "google/flan-t5-base"
        elif config == "large":
            model_name = "google/flan-t5-large"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    if task == "na":
        input_length = 32
        min_new_tokens = 256
        max_new_tokens = 256
    elif task == "text_classification":
        input_length = 32
        min_new_tokens = 0
        max_new_tokens = 5
    elif task == "text_summarization":
        # TODO: hangs with 256
        input_length = 64
        min_new_tokens = 0
        # TODO: issue with generating more than 63 tokens, rotating past cache
        max_new_tokens = 63

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # Set model configurations
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    max_length = config_dict["n_positions"]
    config = T5Config(**config_dict)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    num_blocks = len(model.decoder.block)
    for i in range(num_blocks):
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [1, 1])
        pybuda.config.override_op_size(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [1, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.k.weight_cache_nop", [15, 1])
        pybuda.config.override_t_stream_shape(f"t5.decoder.block.{i}.layer.0.SelfAttention.v.weight_cache_nop", [15, 1])

    encoder_module = pybuda.PyTorchModule("T5_encoder", T5_encoder(model))
    decoder_module_cross_attention = pybuda.PyTorchModule("T5_decoder_with_ca", T5_decoder(model))
    decoder_module_no_cross_attention = pybuda.PyTorchModule("T5_decoder_no_ca", T5_decoder(model))

    def preprocessing(input_text, tokenizer=tokenizer, input_length=input_length):
        encoder_inputs = tokenizer(
            input_text, return_tensors="pt", max_length=input_length, pad_to_max_length=True, truncation=True
        )
        input_ids = encoder_inputs["input_ids"].int()
        encoder_attention_mask = encoder_inputs["attention_mask"].float()

        encoder_last_hidden_state_shape = (1, input_length, config.d_model)
        encoder_last_hidden_state = torch.zeros(encoder_last_hidden_state_shape)
        decoder_input_ids = torch.zeros((1, TILE_DIM), dtype=torch.int)
        decoder_attention_mask = torch.zeros((1, max_length))

        enc_past_cache_self_shape = (1, config.num_heads, max_length - 32, config.d_kv)
        enc_past_cache_cross_shape = (1, 1, 1, 1)
        decoder_ca_inputs = [
            decoder_input_ids,
            decoder_attention_mask,
            encoder_last_hidden_state,
            encoder_attention_mask,
        ]
        for _ in range(num_blocks):
            decoder_ca_inputs += [
                torch.zeros(enc_past_cache_self_shape),
                torch.zeros(enc_past_cache_self_shape),
                torch.zeros(enc_past_cache_cross_shape),
                torch.zeros(enc_past_cache_cross_shape),
            ]

        enc_past_cache_cross_shape = (1, config.num_heads, input_length, config.d_kv)
        decoder_no_ca_inputs = [
            decoder_input_ids,
            decoder_attention_mask,
            encoder_last_hidden_state,
            encoder_attention_mask,
        ]
        for _ in range(num_blocks):
            decoder_no_ca_inputs += [
                torch.zeros(enc_past_cache_self_shape),
                torch.zeros(enc_past_cache_self_shape),
                torch.zeros(enc_past_cache_cross_shape),
                torch.zeros(enc_past_cache_cross_shape),
            ]
        return (
            (input_ids, encoder_attention_mask),
            (decoder_ca_inputs),
            (decoder_no_ca_inputs),
        )

    def preprocessed_collate_fn(batch):
        # Separate inputs and labels
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return inputs, labels

    def forward_wrapper(
        batch,
        output_q,
        device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ):
        # TODO: handle batch input correctly, currently batch-1 only
        inputs = batch[0]
        # unpack preprocessed inputs
        (
            (input_ids, encoder_attention_mask),
            (decoder_ca_inputs),
            (decoder_no_ca_inputs),
        ) = inputs

        device.set_active_subgraph(0)
        device.push_to_inputs((input_ids, encoder_attention_mask))
        pybuda.run_forward()
        ans = output_q.get()
        encoder_last_hidden_state = ans[0].value().detach()
        first_current_index = max_length - TILE_DIM

        decoder_input_ids = torch.zeros((1, TILE_DIM), dtype=torch.int)
        decoder_attention_mask = torch.zeros((1, max_length))

        decoder_attention_mask[0, first_current_index] = 1
        generated_tokens = []
        current_token_index = 0

        for itok in range(max_new_tokens):
            if current_token_index == 1:
                pass
            if current_token_index == 0:
                device.set_active_subgraph(1)
                generate_inputs = (
                    decoder_input_ids,
                    decoder_attention_mask,
                    encoder_last_hidden_state,
                    encoder_attention_mask,
                )
                device.push_to_inputs(generate_inputs)
                pybuda.run_generate(input_count=1, write_index=0)
                ans = output_q.get()
            else:
                device.set_active_subgraph(2)
                generate_inputs = (decoder_input_ids, decoder_attention_mask, encoder_attention_mask)
                device.push_to_inputs(generate_inputs)
                pybuda.run_generate(input_count=1, write_index=0)
                ans = output_q.get()

            lm_head_out = ans[0].value().detach()
            next_token = torch.argmax(lm_head_out[0, current_token_index % TILE_DIM])
            generated_tokens.append(next_token.item())
            current_token_index += 1
            # early stopping condition
            if next_token == eos_token_id and current_token_index >= min_new_tokens:
                break

            if current_token_index % TILE_DIM == 0:
                past_cache_pages = current_token_index // TILE_DIM
                # after one page of past cache, we have to rotate.
                device.set_active_subgraph(3)
                pybuda.run_generate(input_count=0, write_index=0)

                pages_current = 1
                decoder_attention_mask[0, -(past_cache_pages + pages_current) * TILE_DIM :] = 1
                decoder_attention_mask[0, first_current_index:] = 0
                decoder_input_ids[0, :] = tokenizer.pad_token_id

            decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
            decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1

        batch_output = [generated_tokens]
        return batch_output

    # Create model device placement map
    if device == "tt":
        model = {
            "device": [encoder_module, decoder_module_cross_attention, decoder_module_no_cross_attention],
            "forward_wrapper": forward_wrapper,
            "tokenizer": tokenizer,
        }

    # Task specific configuration
    if task == "na":
        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="translate English to German: The house is wonderful.",
            answer="Das Haus ist wunderbar.",
        )
        # preprocess
        dataset.data = [(preprocessing(input_text), label) for (input_text, label) in dataset.data]

        def eval_fn(outputs, labels):
            return 0

    elif task == "text_classification":
        # Load SST-2 Dataset
        sst2_dataset = load_dataset("glue", "sst2", split="validation")
        dataset = PipelineDataset(
            dataset=sst2_dataset,
            input_text="sentence",
            label="label",
            prepend_text="Is the following review positive or negative: ",
        )
        # preprocess
        dataset.data = [(preprocessing(input_text), label) for (input_text, label) in dataset.data]

        # Define evaluation function
        def eval_fn(outputs, labels, tokenizer=tokenizer):
            import evaluate

            accuracy_metric = evaluate.load("accuracy")
            pred_labels = []
            true_labels = []
            for b_out in outputs:
                pred_b_out = [1 if "positive" in tokenizer.decode(item) else 0 for item in b_out]
                pred_labels.extend(pred_b_out)

            for label in labels:
                true_labels.extend(label)
            eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)

            return eval_score["accuracy"]

    elif task == "text_summarization":

        # Load CNN / DailyMail dataset
        cnndm_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:20]")
        dataset = PipelineDataset(
            dataset=cnndm_dataset,
            input_text="article",
            label="highlights",
            prepend_text="summarize: ",
        )

        # preprocess
        dataset.data = [(preprocessing(input_text), label) for (input_text, label) in dataset.data]

        # Define evaluation function
        def eval_fn(outputs, labels):
            import evaluate

            rouge_metric = evaluate.load("rouge")
            pred_labels = []
            true_labels = []
            for b_out in outputs:
                pred_b_out = [tokenizer.decode(item) for item in b_out]
                pred_labels.extend(pred_b_out)

            for label in labels:
                true_labels.extend(label)
            eval_score = rouge_metric.compute(references=true_labels, predictions=pred_labels)
            print(eval_score)

            return eval_score["rouge1"]

    # create compile_input
    # NOTE: in the general case compile input may look different from the per sample
    # required input. This can be done more efficiently and much processing can be avoided.
    compile_sentance = "translate English to German: The house is wonderful."
    compile_inputs = preprocessing(compile_sentance)
    model["compile_inputs"] = compile_inputs

    # Create DataLoader
    generator = DataLoader(
        dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=preprocessed_collate_fn
    )

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
