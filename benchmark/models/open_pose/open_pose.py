# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.models.open_pose.utils.common import openpose_preprocess, run_coco_eval, to_coco_json_format_batch
from benchmark.models.open_pose.utils.pose_extractor import recalc_pose

from ...common import COCODataset, DummyCVDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["2d", "3d"])
def open_pose(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):
    if device == "tt":
        import pybuda

        # Configurations
        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_auto_transposing_placement = True

        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"
            os.environ["PYBUDA_RIBBON2"] = "1"

        os.environ["PYBUDA_SUPRESS_T_FACTOR_MM"] = "13"

        # These are about to be enabled by default.
        #
        if data_type != "Bfp8_b":
            os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
        else:
            # tenstorrent/pybuda#2228
            os.environ["PYBUDA_LEGACY_KERNEL_BROADCAST"] = "1"

        os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"

    # Set model parameters based on chosen task and model configuration
    model_name = ""
    img_res = 224
    if config == "2d":
        model_name = "lwopenpose2d_mobilenet_cmupan_coco"
    elif config == "3d":
        model_name = "lwopenpose3d_mobilenet_cmupan_coco"
    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = 32  # default

    if task == "na":
        # Load model
        model = ptcv_get_model(model_name, pretrained=True)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_open_pose_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)

        collate_fn = None

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "pose_estimation":
        # params
        # increase n_samples to 5000 evaluate full COCO val2017 dataset
        n_samples = 128
        # res sizes: [(224,224), (368, 368),], default is 224x224
        height, width = img_res, img_res
        assert (height, width) in [
            (224, 224),
            (368, 368),
        ]
        split = "val2017"
        # Load model
        model = ptcv_get_model(model_name, pretrained=True)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_open_pose_{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # load COCO dataset
        data_dir = Path(os.environ.get("MLDATA_DIR", f"{Path.home()}/.cache/mldata"), "coco")
        ann_type = "keypoints"

        dataset = COCODataset(
            data_dir=data_dir,
            split=split,
            ann_type=ann_type,
            n_samples=n_samples,
        )
        dataset.data = openpose_preprocess(dataset, target_height=height, target_width=width)

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
            from statistics import mean

            def divide_chunks(input_list, number_of_samples):
                for i in range(0, len(input_list), number_of_samples):
                    yield input_list[i : i + number_of_samples]

            # Chunk the outputs in case there are multiple loop counts
            outputs_chunked = list(divide_chunks(outputs, n_samples // microbatch))
            labels_chunked = list(divide_chunks(labels, n_samples // microbatch))

            m_ap = []
            for outputs, labels in tqdm(
                zip(outputs_chunked, labels_chunked), total=len(outputs_chunked), desc="loop chunks"
            ):
                coco_res = []
                full_label_img_ids = [int(lbl[0]) for batch_labels in labels for lbl in batch_labels]
                for batch_pred, batch_label in tqdm(zip(outputs, labels), total=len(outputs), desc="batches"):
                    if isinstance(batch_pred, list):
                        batch_pred = (
                            batch_pred[0].to_pytorch().detach().double().numpy()
                            if not isinstance(batch_pred[0], torch.Tensor)
                            else batch_pred[0].detach().double().numpy()
                        )
                    else:
                        # handle cpu and cuda devices
                        batch_pred = batch_pred.to("cpu").detach().double().numpy()
                    batch_label = torch.stack(batch_label).numpy()
                    coco_keypoints, pred_score, img_ids = recalc_pose(batch_pred, batch_label)
                    coco_res.extend(to_coco_json_format_batch(coco_keypoints, pred_score, img_ids))

                m_ap.append(run_coco_eval(coco_res, full_label_img_ids, ann_type, data_dir, split=split))
            return mean(m_ap)

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=collate_fn)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
