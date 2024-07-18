# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...common import BrainSegmentationDataset, DummyCVDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["256"])
def unet(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":

        import pybuda
        from pybuda._C.backend_api import BackendDevice

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_tvm_constant_prop = True
        compiler_cfg.enable_auto_transposing_placement = True

        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"

        if data_type == "Bfp8_b" and pybuda.detect_available_devices()[0] == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_ENABLE_DRAM_IO_BUFFER_SCALING"] = "1"
            os.environ["PYBUDA_ENABLE_INPUT_BUFFER_SCALING_FOR_NOC_READERS"] = "1"

        # Manually enable amp light for Ribbon
        if compiler_cfg.balancer_policy == "Ribbon":
            compiler_cfg.enable_amp_light()

        os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"
        os.environ["PYBUDA_ALLOW_MULTICOLUMN_SPARSE_MATMUL"] = "1"
        os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "60"

        # These are about to be enabled by default.
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

    # Set model parameters based on chosen task and model configuration
    if config == "256":
        model = torch.hub.load(
            "mateuszbuda/brain-segmentation-pytorch",
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )
        img_res = 256
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    if task == "na":
        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_unet_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)

        collate_fn = None

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "segmentation":
        n_samples = 240
        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_unet_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        data_dir = Path(os.environ.get("MLDATA_DIR", f"{Path.home()}/.cache/mldata"), "lgg_segmentation")
        dataset = BrainSegmentationDataset(
            data_dir=data_dir,
            n_samples=n_samples,
        )

        def collate_fn(batch):
            # Separate inputs and labels
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            # Stack inputs and labels into two separate tensors
            inputs = [torch.stack(inputs)]
            labels = torch.stack(labels)
            return inputs, labels

        # Define evaluation function
        def eval_fn(outputs, labels):
            from torchmetrics import AveragePrecision

            ap_list = []
            ap_metric = AveragePrecision(task="binary", thresholds=torch.arange(0.5, 1.0, 0.05))
            for batch_outputs, batch_labels in tqdm(zip(outputs, labels), total=len(outputs)):
                if isinstance(batch_outputs, list):
                    batch_pred = (
                        batch_outputs[0].to_pytorch().detach()
                        if not isinstance(batch_outputs[0], torch.Tensor)
                        else batch_outputs[0].detach()
                    )
                else:
                    # handle cpu and cuda devices
                    batch_pred = batch_outputs.to("cpu").detach()
                labels_bin = batch_labels == 255
                for pred, label in zip(batch_pred, labels_bin):
                    ap = ap_metric(pred, label)
                    ap_list.append(ap)

            m_ap = torch.stack(ap_list).mean().item()
            return m_ap

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
