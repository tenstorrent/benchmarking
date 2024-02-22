# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from ...common import (
    DummyCVDataset,
    ImageNetDataset,
    benchmark_model,
    torch_df_from_str,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


@benchmark_model(configs=["wide_resnet50_2", "wide_resnet101_2"])
def wideresnet(
    training: bool, task: str, config: str, microbatch: int, device: str, data_type: str
):
    if device == "tt":
        import pybuda

        # Configurations
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_t_streaming = True
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    target_microbatch = 64

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    if task == "na":
        # Load model
        model = torch.hub.load("pytorch/vision:v0.10.0", config, pretrained=True)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(
            microbatch=microbatch,
            channels=3,
            height=224,
            width=224,
            data_type=data_type,
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "image_classification":
        # Load model
        model = torch.hub.load("pytorch/vision:v0.10.0", config, pretrained=True)

        version = "torch"
        feature_extractor = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        imagenet_dataset = load_dataset(
            "imagenet-1k", split="validation", use_auth_token=True, streaming=True
        )

        dataset_iter = iter(imagenet_dataset)
        dataset = []

        for _ in range(1000):
            dataset.append(next(dataset_iter))
        dataset = ImageNetDataset(
            dataset=dataset, feature_extractor=feature_extractor, version=version
        )

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

            eval_score = accuracy_metric.compute(
                references=true_labels, predictions=pred_labels
            )

            return eval_score["accuracy"]

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(
        dataset, batch_size=microbatch, shuffle=False, drop_last=True
    )

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn