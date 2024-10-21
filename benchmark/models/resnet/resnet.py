# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from torchvision import transforms
import onnx

from ...common import DummyCVDataset, ImageNetDataset, benchmark_model, torch_df_from_str

@benchmark_model(configs=["resnet18", "resnet50","resnet50_onnx"])
def resnet(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    script_dir = os.path.dirname(os.path.realpath(__file__))
    if device == "tt":
        import pybuda
        from pybuda._C.backend_api import BackendDevice

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_auto_transposing_placement = True

        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"

        os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"
        os.environ["PYBUDA_ALLOW_MULTICOLUMN_SPARSE_MATMUL"] = "1"

        if data_type == "Bfp8_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
            os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"
            # Enable Data Movement Estimates
            os.environ["PYBUDA_BALANCER_USE_DRAM_BW_ESTIMATES"] = "1"
            os.environ["PYBUDA_BALANCER_USE_NOC_BW_ESTIMATES"] = "1"

        # These are about to be enabled by default.
        #
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

        if data_type == "Fp16_b":
            os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES_APPLY_FILTERING"] = "1"

        if data_type == "Bfp8_b":
            pybuda.config.configure_mixed_precision(name_regex="input.*add.*", output_df=pybuda.DataFormat.Float16_b)

    # Set model parameters based on chosen task and model configuration
    if config == "resnet18":
        model_name = "microsoft/resnet-18"
        target_microbatch = 64
    elif config == "resnet50":
        model_name = "microsoft/resnet-50"
        target_microbatch = 64
    elif config == "resnet50_onnx":
        target_microbatch = 64
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    print(script_dir)
    load_local_path = script_dir + "/resnet50.onnx"

    # Task specific configuration
    if task == "na":

        # Load model
        if config == "resnet50_onnx":
            model = onnx.load(load_local_path)
        else:
            model = ResNetForImageClassification.from_pretrained(model_name)

            # Configure model mode for training or evaluation
            if training:
                model.train()
            else:
                model.eval()

        # Create model device placement map
        if device == "tt":
            if config == "resnet50_onnx":
                model = {"device":pybuda.OnnxModule(f"on_{config}", model, load_local_path)}
            else:
                model = {"device": pybuda.PyTorchModule(f"pt_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=224, width=224, data_type=data_type)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "image_classification":

        # Load model & feature extractor
        if config == "resnet50_onnx":
            model = onnx.load(load_local_path)
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        else:
            model = ResNetForImageClassification.from_pretrained(model_name)
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            # Configure model mode for training or evaluation
            if training:
                model.train()
            else:
                model.eval()

        # Create model device placement map
        if device == "tt":
            if config == "resnet50_onnx":
                model = {"device":pybuda.OnnxModule(f"on_{config}", model, load_local_path)}
            else:
                model = {"device": pybuda.PyTorchModule(f"pt_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        imagenet_dataset = load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True)
        dataset_iter = iter(imagenet_dataset)
        dataset = []
        for _ in range(1000):
            dataset.append(next(dataset_iter))
        dataset = ImageNetDataset(dataset=dataset, feature_extractor=feature_extractor)

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
                    output = output[0].detach().cpu()
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
