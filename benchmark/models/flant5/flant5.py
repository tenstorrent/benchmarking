# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Config, pipeline

from ...common import DummyPipelineDataset, PipelineDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["small", "base", "large"])
def flant5(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda._C.backend_api import BackendDevice
        from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

        os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
        os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
        os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"
        
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_tvm_cpu_fallback = False
        compiler_cfg.default_df_override = pybuda._C.Float16_b
        compiler_cfg.default_dram_parameters = False
        if pybuda.detect_available_devices()[0] == BackendDevice.Grayskull:
            compiler_cfg.enable_auto_fusing = False
        compiler_cfg.enable_amp_light()
        # compiler_cfg.compile_subgraphs = True
        # compiler_cfg.enable_link_past_cache_ios = True

        # os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        # os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
        # os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        # os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
        # os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"
        
        # compiler_cfg = pybuda.config._get_global_compiler_config()
        # compiler_cfg.enable_tvm_cpu_fallback = False
        # compiler_cfg.default_df_override = pybuda._C.Float16_b
        # compiler_cfg.default_dram_parameters = False
        # compiler_cfg.enable_amp_light()
        # compiler_cfg.compile_subgraphs = True
        # compiler_cfg.enable_link_past_cache_ios = True

        # # Add PyBUDA configurations
        # os.environ["PYBUDA_PAD_OUTPUT_BUFFER"] = "1"
        # os.environ["TT_BACKEND_MULTI_THREADED_PUSH"] = "1"
        # os.environ["PYBUDA_DISABLE_DYNAMIC_DRAM"] = "1"
        # os.environ["PYBUDA_EXTRA_L1_MARGIN"] = "120000"
        # os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"
        # os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "35000"
        # os.environ["TT_BACKEND_DRAM_POLLING_FREQUENCY"] = "64"
        # os.environ["TT_BACKEND_PROFILER"] = "1"
        # os.environ["TT_BACKEND_EPOCH_BIN_NUM_SLOTS"] = "64"
        # os.environ["PYBUDA_ROTATE_PAST_CACHE_PARAMS"] = "1"
        # os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"

        # compiler_cfg = pybuda.config._get_global_compiler_config()
        # compiler_cfg.enable_t_streaming = True
        # compiler_cfg.enable_tvm_cpu_fallback = False
        # compiler_cfg.default_df_override = pybuda._C.Float16_b
        # compiler_cfg.default_dram_parameters = False
        # compiler_cfg.enable_auto_fusing = False
        # compiler_cfg.enable_amp_light()

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
        # TODO: same as t5_past_cache_enc_dec implementation
        input_length = 64
        min_new_tokens = 0
        # TODO: same as t5_past_cache_enc_dec implementation
        max_new_tokens = 63

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # Set model configurations
    config = T5Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = T5Config(**config_dict)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=input_length, padding="max_length", truncation=True
    )

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    # Create model device placement map
    if device == "tt":
        model = pybuda_pipeline(
            "text2text-generation", model=model, tokenizer=tokenizer, pybuda_max_length=input_length
        )
    else:
        device = 0 if device == "cuda" else -1
        model = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_df_from_str(data_type),
        )

    # set model parameters
    model.model.config.early_stopping = True
    model.model.config.length_penalty = 1.0
    # disable ngram repeat restriction to not disadvantage other implementations
    model.model.config.no_repeat_ngram_size = 0
    model.model.config.max_length = max_new_tokens
    model.model.config.min_length = min_new_tokens
    model.model.config.num_beams = 1
    model.model.config.num_return_sequences = 1
    # set key for accessing output text
    model.output_key = "generated_text"

    # Task specific configuration
    if task == "na":

        # set task specific model parameters
        model.model.config.early_stopping = False

        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="translate English to German: The house is wonderful.",
            answer="Das Haus ist wunderbar.",
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "text_classification":

        # Load SST-2 Dataset
        sst2_dataset = load_dataset("glue", "sst2", split="validation[:100]")
        dataset = PipelineDataset(
            dataset=sst2_dataset,
            input_text="sentence",
            label="label",
            prepend_text="Is the following review positive or negative: ",
        )

        # Define evaluation function
        def eval_fn(outputs, labels):
            import evaluate

            accuracy_metric = evaluate.load("accuracy")
            pred_labels = []
            true_labels = []
            for batch in outputs:
                for item in batch:
                    output = item["generated_text"]
                    pred_labels.extend([1 if "positive" in output else 0])
            for label in labels:
                true_labels.extend(label)
            eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)

            return eval_score["accuracy"]

    elif task == "text_summarization":

        # Load CNN / DailyMail dataset
        cnndm_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10]")
        dataset = PipelineDataset(
            dataset=cnndm_dataset,
            input_text="article",
            label="highlights",
            prepend_text="summarize: ",
        )

        # Define evaluation function
        def eval_fn(outputs, labels):
            import evaluate

            rouge_metric = evaluate.load("rouge")
            pred_labels = []
            true_labels = []
            for batch in outputs:
                for item in batch:
                    output = item["generated_text"]
                    pred_labels.extend([output])
            for label in labels:
                true_labels.extend(label)
            eval_score = rouge_metric.compute(references=true_labels, predictions=pred_labels)
            print(eval_score)
            return eval_score["rouge1"]

    # NOTE: must pre-truncate the dataset before passing to pipeline, no easy way to
    # make pipeline.__call__ method truncate all input samples
    # truncation is needed to compare same input as in past_cache_encoder_decoder
    dataset = [
        (tokenizer.decode(tokenizer(input, truncation=True, padding=False, max_length=input_length).input_ids), label)
        for (input, label) in dataset
    ]
    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
