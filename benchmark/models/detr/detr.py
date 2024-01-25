# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection

from ...common import DummyCVDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["resnet50", "resnet101"])
def detr(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

        os.environ["PYBUDA_ENABLE_STABLE_SOFTMAX"] = "1"
        os.environ["PYBUDA_ENABLE_BROADCAST_SPLITTING"] = "1"
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.retain_tvm_python_files = True
        compiler_cfg.enable_tvm_constant_prop = True
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True
        compiler_cfg.verify_pybuda_codegen_vs_framework = False  # hacking 7x7 to 1x1 will cause mismatches

    # Set model parameters based on chosen task and model configuration
    img_res = 224
    target_microbatch = 32

    if config == "resnet50":
        model_name = "facebook/detr-resnet-50"
    elif config == "resnet101":
        model_name = "facebook/detr-resnet-101"
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Load model
        model = DetrForObjectDetection.from_pretrained(model_name)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_detr_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
