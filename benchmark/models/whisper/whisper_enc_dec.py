# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader

from benchmark.models.whisper.whisper_impl import generate_model_whisper_enc_dec

from ...common import DummyPipelineDataset, LibriSpeechDataset, benchmark_model


@benchmark_model(configs=["tiny", "base", "small", "medium", "large"])
def whisper_enc_dec(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):
    assert device == "tt", "This model is only supported on TT hardware"

    import pybuda
    from pybuda._C.backend_api import BackendDevice

    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.dont_fuse("subtract_634")

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"

    if data_type == "Fp16_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
    
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            pybuda.config.set_epoch_break("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
            pybuda.config.override_op_size("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 12))

    # if compiler_cfg.balancer_policy == "default":
    #     compiler_cfg.balancer_policy = "Ribbon"
    #     os.environ["PYBUDA_RIBBON2"] = "1"

    # if data_type == "Fp16_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
    #     os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
    #     os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
    
    # available_devices = pybuda.detect_available_devices()
    # if available_devices:
    #     if available_devices[0] == BackendDevice.Grayskull:
    #         pybuda.config.set_epoch_break("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2")
    #         pybuda.config.override_op_size("conv2d_9.dc.sparse_matmul.9.dc.sparse_matmul.1.lc2", (1, 12))

    # Determine model variant
    if config == "small":
        variant = "openai/whisper-small"
    else:
        raise RuntimeError("Unknown config")

    # Set model parameters based on chosen task and model configuration
    if task == "na" or task == "asr":
        if config == "tiny":
            model_name = "openai/whisper-tiny"
            raise NotImplementedError("This model implementation does not support: openai/whisper-tiny")
        elif config == "base":
            model_name = "openai/whisper-base"
        elif config == "small":
            model_name = "openai/whisper-small"
        elif config == "medium":
            model_name = "openai/whisper-medium"
        elif config == "large":
            model_name = "openai/whisper-large-v2"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # generate model
    modules, other = generate_model_whisper_enc_dec(model_name)
    first_current_index = other["first_current_index"]
    processor = other["processor"]
    embed_positions_weight = other["embed_positions_weight"]
    logits_processor = other["logits_processor"]
    max_length = other["max_length"]

    if task == "na":
        min_new_tokens = 69
        max_new_tokens = 69
    elif task == "asr":
        min_new_tokens = 0
        max_new_tokens = 256

    def preprocess_func(
        input_arr,
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        prefix_tokens=processor.get_decoder_prompt_ids(language="english", task="transcribe"),
        first_current_index=first_current_index,
        pad_model=True,
        max_length=max_length,
    ):
        from pybuda.pybudaglobal import TILE_DIM

        inputs = feature_extractor(input_arr, return_tensors="pt", sampling_rate=16000)

        if pad_model:
            input_features = torch.nn.functional.pad(inputs.input_features, (0, 72, 0, 0))
        else:
            input_features = inputs.input_features

        decoder_input_ids = torch.ones((1, TILE_DIM), dtype=torch.int) * tokenizer.pad_token_id
        decoder_attention_mask = torch.zeros((1, max_length))
        decoder_input_ids[0, 0] = tokenizer.encode("<|startoftranscript|>")[0]
        decoder_attention_mask[0, first_current_index] = 1
        current_token_index = 0

        for idx, token in prefix_tokens:
            decoder_input_ids[0, idx] = token
            decoder_attention_mask[0, first_current_index + idx] = 1
            current_token_index = idx

        # encoder hangs for some variants, for now run on cpu
        # encoder_last_hidden_state = encoder(input_features)[0].detach()

        return (input_features, decoder_input_ids, decoder_attention_mask, current_token_index)

    def forward_wrapper(
        batch,
        output_q,
        device,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        first_current_index=first_current_index,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        logits_processor=logits_processor,
        embed_positions_weight=embed_positions_weight,
        max_length=max_length,
    ):
        from pybuda.pybudaglobal import TILE_DIM

        # TODO: handle batch input correctly, currently batch-1 only
        inputs = batch[0]
        # unpack input
        (input_features, decoder_input_ids, decoder_attention_mask, current_token_index) = preprocess_func(inputs)

        active_subgraph = 0
        device.set_active_subgraph(active_subgraph)
        device.push_to_inputs((input_features,))
        pybuda.run_forward()
        ans = output_q.get()
        encoder_last_hidden_state = ans[0].value().detach()
        active_subgraph = 1
        generated_tokens = []
        encoder_last_hidden_state_consumed = False
        position_ids = torch.arange(32, dtype=torch.long)
        position_embeds = embed_positions_weight[position_ids]

        for _ in range(max_new_tokens):
            if not encoder_last_hidden_state_consumed:
                encoder_last_hidden_state_consumed = True
                device.set_active_subgraph(active_subgraph)
                generate_inputs = (
                    decoder_input_ids,
                    decoder_attention_mask,
                    encoder_last_hidden_state,
                    position_embeds,
                )
                device.push_to_inputs(generate_inputs)
                pybuda.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
                ans = output_q.get()
                device.set_active_subgraph(active_subgraph + 1)
            else:
                generate_inputs = (decoder_input_ids, decoder_attention_mask, position_embeds)
                device.push_to_inputs(generate_inputs)
                pybuda.run_generate(input_count=1, write_index=current_token_index // TILE_DIM)
                ans = output_q.get()
            lm_head_out = ans[0].value().detach()
            scores = logits_processor(
                decoder_input_ids[:, :current_token_index], lm_head_out[:, current_token_index % TILE_DIM]
            )
            next_token = torch.argmax(scores, dim=-1).item()
            generated_tokens.append(next_token)
            current_token_index += 1
            # early stopping condition
            if next_token == eos_token_id and current_token_index >= min_new_tokens:
                break

            if current_token_index % TILE_DIM == 0 and current_token_index != max_length:
                position_ids = position_ids + TILE_DIM
                position_embeds = embed_positions_weight[position_ids]
                decoder_attention_mask[0, :current_token_index] = 1
                decoder_attention_mask[0, first_current_index:] = 0
                decoder_input_ids[0, :] = pad_token_id
            decoder_input_ids[0, current_token_index % TILE_DIM] = next_token
            decoder_attention_mask[0, first_current_index + (current_token_index % TILE_DIM)] = 1

        batch_output = [generated_tokens]
        return batch_output

    # Create model device placement map
    model = {
        "device": modules,
        "forward_wrapper": forward_wrapper,
        "compile_inputs": other["compile_inputs"],
        "tokenizer": processor.tokenizer,
        "verify_cfg": pybuda.verify.VerifyConfig(
            verify_pybuda_codegen_vs_framework=True,
            enabled=False,
        ),
    }

    # Task specific configuration
    if task == "na":
        import librosa

        # Get sample
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample_audio = ds[0]["audio"]["path"]

        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text=sample_audio,
            answer="",
        )
        # need to load audio files with sample_rate=16000 for whisper
        dataset.data = [(librosa.load(d, sr=16000)[0], label) for d, label in dataset.data]
        collate_fn = None

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "asr":
        # for full validation `librispeech_asr` split="test.clean"
        # contains 2620 audio samples
        # ds = load_dataset("librispeech_asr", "clean", split="test")
        # for quicker testing `hf-internal-testing/librispeech_asr_dummy`
        # split="validation.clean" contains 73 audio samples
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # create dataset with handling class
        dataset = LibriSpeechDataset(dataset=ds)

        # librispeech_asr has samples of different lengths, collate_fn is required
        def collate_fn(batch):
            # This collate_fn separates the data and labels
            data = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            return data, labels

        # Define evaluation function
        def eval_fn(outputs=None, labels=None, tokenizer=processor.tokenizer, **kwargs):
            pred_output = []
            for b_out in outputs:
                # labels are all upper case, decode and make outputs upper case to match
                pred_b_out = [tokenizer.decode(item, skip_special_tokens=True).strip().upper() for item in b_out]
                pred_output.extend(pred_b_out)
            # labels is lists of lists, flatten them
            # there also may not be "." at the end of each label
            flattened_labels = [lbl if lbl[-1] == "." else lbl + "." for batch_labels in labels for lbl in batch_labels]
            # calculate Word Error Rate (WER)
            # see: https://huggingface.co/spaces/evaluate-metric/wer
            wer = load("wer")
            wer_score = wer.compute(predictions=pred_output, references=flattened_labels)
            # word accuracy score (positive is better) is complement of error score
            # W_acc = 1 - WER = (Correct - Inserted) / (Number of words in the reference)
            eval_score = 1 - wer_score
            return eval_score

    # Create DataLoader
    generator = DataLoader(
        dataset,
        batch_size=microbatch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
