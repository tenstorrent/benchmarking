# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, pipeline

from ...common import DummyPipelineDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["base"])
def gpt2(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

    # Set model parameters based on chosen task and model configuration
    if task == "na":
        if config == "base":
            model_name = "gpt2"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # Task specific configuration
    if task == "na":

        # Set model configurations
        config = GPT2Config.from_pretrained(model_name)
        config_dict = config.to_dict()
        config_dict["return_dict"] = False
        config_dict["use_cache"] = False
        config = GPT2Config(**config_dict)

        # Load model
        model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            device = 0 if device == "cuda" else -1
            model = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch_df_from_str(data_type),
            )

        # set model parameters
        model.model.config.max_length = 10
        model.model.config.min_length = 10
        # set key for accessing output text
        model.output_key = "generated_text"

        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="My name is Bert and I like to",
            answer="My name is Bert and I like to",
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
