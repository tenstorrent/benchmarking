# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.multiprocessing
from datasets import load_dataset
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader
from torchvision import transforms

from ...common import DummyCVDataset, ImageNetDataset, benchmark_model, torch_df_from_str

torch.multiprocessing.set_sharing_strategy("file_system")


@benchmark_model(
    configs=[
        "w18",
        "v2_w18",
        "v2_w30",
        "v2_w32",
        "v2_w40",
        "v2_w44",
        "v2_w48",
        "v2_w64",
    ]
)
def hrnet(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda._C.backend_api import BackendDevice

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_auto_transposing_placement = True

        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"

        if data_type == "Bfp8_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
            os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"

        os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "46" # removing causes hang #2139
        os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

        # These are about to be enabled by default.
        #
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
        if data_type == "Fp16_b":
            # Hangs with autotranspose on #2542
            compiler_cfg.enable_auto_transposing_placement = False

        # Manually enable amp light for Ribbon
        if compiler_cfg.balancer_policy == "Ribbon":
            compiler_cfg.enable_amp_light()

        # compiler_cfg.enable_auto_transposing_placement = True

        # if compiler_cfg.balancer_policy == "default":
        #     compiler_cfg.balancer_policy = "Ribbon"
        #     os.environ["PYBUDA_RIBBON2"] = "1"

        # if data_type == "Bfp8_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
        #     os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
        #     os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"

        # os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "46" # removing causes hang #2139
        # os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

        # # These are about to be enabled by default.
        # #
        # os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
        # if data_type == "Fp16_b":
        #     # Hangs with autotranspose on #2542
        #     compiler_cfg.enable_auto_transposing_placement = False

        # # Manually enable amp light for Ribbon
        # if compiler_cfg.balancer_policy == "Ribbon":
        #     compiler_cfg.enable_amp_light()


    # Set model parameters based on chosen task and model configuration
    img_res = 224
    target_microbatch = 32

    if config == "w18":
        model_name = "hrnet_w18_small_v2"
    elif config == "v2_w18":
        model_name = "hrnetv2_w18"
    elif config == "v2_w30":
        model_name = "hrnetv2_w30"
    elif config == "v2_w32":
        model_name = "hrnetv2_w32"
    elif config == "v2_w40":
        model_name = "hrnetv2_w40"
    elif config == "v2_w44":
        model_name = "hrnetv2_w44"
    elif config == "v2_w48":
        model_name = "hrnetv2_w48"
    elif config == "v2_w64":
        model_name = "hrnetv2_w64"
        if data_type == "Bfp8_b":
            if "TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE" not in os.environ:
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{10*1024}"
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                pybuda.config._internal_insert_fj_buffering_nop('add_312', ['add_341'], nop_count=2)
                pybuda.config.set_epoch_break("resize2d_3176.dc.sparse_matmul.3.lc2")
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Load model
        model = ptcv_get_model(model_name, pretrained=True)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_hrnet_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "image_classification":

        # Load model
        model = ptcv_get_model(model_name, pretrained=True)
        version = "torch"
        feature_exctractor = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_hrnet_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        imagenet_dataset = load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True)
        dataset_iter = iter(imagenet_dataset)
        dataset = []
        for _ in range(1000):
            dataset.append(next(dataset_iter))
        dataset = ImageNetDataset(dataset=dataset, feature_extractor=feature_exctractor, version=version)

        # Define evaluation function
        def eval_fn(outputs, labels):
            import evaluate

            accuracy_metric = evaluate.load("accuracy")
            pred_labels = []
            true_labels = []
            for output in outputs:
                if device == "tt":
                    output = output[0].value()
                else:
                    output = output.detach().cpu()
                pred_labels.extend(torch.argmax(output, axis=-1))
            for label in labels:
                true_labels.extend(label)
            eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)

            return eval_score["accuracy"]

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
