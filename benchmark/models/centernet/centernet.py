# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import onnx
import torch
from torch.utils.data import DataLoader

from benchmark.models.yolo_v5.utils.common import coco_post_process_bbox, run_coco_eval, yolov5_preprocessing
from ...common import COCODataset, DummyCVDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["onnx"])
def centernet(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

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
            # Enable Data Movement Estimates
            os.environ["PYBUDA_BALANCER_USE_DRAM_BW_ESTIMATES"] = "1"
            os.environ["PYBUDA_BALANCER_USE_NOC_BW_ESTIMATES"] = "1"

        os.environ["PYBUDA_ENABLE_HOST_INPUT_NOP_BUFFERING"] = "1"

        # These are about to be enabled by default.
        os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"

        if data_type == "Fp16_b":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

        if data_type == "Bfp8_b":
            pybuda.config.configure_mixed_precision(name_regex="input.*add.*", output_df=pybuda.DataFormat.Float16_b)
            pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
            pybuda.config.configure_mixed_precision(op_type="multiply", math_fidelity=pybuda.MathFidelity.HiFi2)
            pybuda.config.configure_mixed_precision(op_type="matmul", math_fidelity=pybuda.MathFidelity.HiFi2)

    # Set model parameters based on chosen task and model configuration
    if config == "onnx":
        model_name = os.path.dirname(os.path.realpath(__file__)) + "/centernet.onnx"
        target_microbatch = 4
        img_res = 512
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration

    if task == "na":
        if config == "onnx":
            model = onnx.load(model_name)
            if device == "tt":
                model = {"device": pybuda.OnnxModule(f"onnx_centernet", model,model_name)}
            else:
                model = model.to(device, dtype=torch_df_from_str(data_type))     
            dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)
        else:
            raise RuntimeError("Unknown config")

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0
    elif task == "object_detection":
        # params
        # increase n_samples to 5000 evaluate full COCO val2017 dataset
        n_samples = 1024
        height, width = img_res, img_res
        split = "val2017"

        # Configure model params for training or evaluation


        # Create model device placement map
        
        if config == "onnx":
            if training:
                raise RuntimeError("training not supported")
            model = onnx.load(model_name)
            if device == "tt":
                model = {"device": pybuda.OnnxModule(f"onnx_centernet", model,model_name)}
            else:
                if training:
                    model.train()
                else:
                    model.eval()
                model = model.to(device, dtype=torch_df_from_str(data_type))     
        else:
            raise RuntimeError("Unknown config")
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
            return inputs,  labels


        ## put evaluation function as you want
        '''
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
        '''
        def eval_fn(**kwargs):
            return 0.0

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
