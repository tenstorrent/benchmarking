# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark.models.yolo_v5.utils.common import coco_post_process_bbox, run_coco_eval, yolov5_preprocessing

from ...common import COCODataset, DummyCVDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["s"])
def yolo_v5(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda._C.backend_api import BackendDevice

        compiler_cfg = pybuda.config._get_global_compiler_config()

        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"
            os.environ["PYBUDA_RIBBON2"] = "1"

            # These are about to be enabled by default.
            #
            os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
            if data_type != "Bfp8_b":
                os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

            if data_type == "Bfp8_b":
                os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
                os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
                os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

        available_devices = pybuda.detect_available_devices()
        if available_devices[0] == BackendDevice.Grayskull:
            compiler_cfg.enable_tm_cpu_fallback = True
            compiler_cfg.enable_auto_fusing = False  # required to fix accuracy
            os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
        elif available_devices[0] == BackendDevice.Wormhole_B0:
            os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "49"

    # Set model parameters based on chosen task and model configuration
    if config == "s":
        # Load model
        model = model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        target_microbatch = 32
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_yolov5_s_hf_1", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=320, width=320, data_type=data_type)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

        collate_fn = None

    elif task == "object_detection":
        # params
        # increase n_samples to 5000 evaluate full COCO val2017 dataset
        n_samples = 1024
        # res sizes: [(640,640), (480,480), (1280,1280)]
        height, width = 320, 320
        split = "val2017"

        # Configure model params for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_yolov5_s_hf_1", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # load COCO dataset
        data_dir = Path(os.environ.get("MLDATA_DIR", f"{Path.home()}/.cache/mldata"), "coco")
        ann_type = "bbox"

        dataset = COCODataset(
            data_dir=data_dir,
            split=split,
            ann_type=ann_type,
            n_samples=n_samples,
        )
        dataset.data = yolov5_preprocessing(dataset, target_height=height, target_width=width)

        def collate_fn(batch):
            # Separate inputs and labels
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            # Stack inputs
            inputs = torch.stack(inputs)
            # add list wrapper for CPU batch shape compatibility
            inputs = [inputs]
            return inputs, labels

        # Define evaluation function
        def eval_fn(
            outputs,
            labels,
            data_dir=data_dir,
            split=split,
            ann_type=ann_type,
        ):
            coco_res = coco_post_process_bbox(outputs, labels)
            # each annotation is a list per image, so we can take first element
            image_ids = [lbl["image_id"] for batch_labels in labels for lbl in batch_labels]
            m_ap = run_coco_eval(coco_res, image_ids, ann_type, data_dir, split)
            return m_ap

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
