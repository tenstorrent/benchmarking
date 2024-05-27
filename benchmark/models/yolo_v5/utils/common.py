# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
import torch
import yolov5
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov5.models.common import Detections
from yolov5.utils.dataloaders import exif_transpose, letterbox
from yolov5.utils.general import Profile, coco80_to_coco91_class, non_max_suppression, scale_boxes, xyxy2xywh


def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    if not isinstance(ims, (list, tuple)):
        ims = [ims]
    num_images = len(ims)
    shape_orig, shape_infer, filenames = [], [], []

    for idx, img in enumerate(ims):
        filename = getattr(img, "filename", f"image{idx}")
        img = np.asarray(exif_transpose(img))
        filename = Path(filename).with_suffix(".jpg").name
        filenames.append(filename)

        if img.shape[0] < 5:
            img = img.transpose((1, 2, 0))

        if img.ndim == 3:
            img = img[..., :3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        shape_orig.append(img.shape[:2])
        scale = max(size) / max(img.shape[:2])
        shape_infer.append([int(dim * scale) for dim in img.shape[:2]])
        ims[idx] = img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)

    shape_infer = [size[0] for _ in np.array(shape_infer).max(0)]
    imgs_padded = [letterbox(img, shape_infer, auto=False)[0] for img in ims]
    imgs_padded = np.ascontiguousarray(np.array(imgs_padded).transpose((0, 3, 1, 2)))
    tensor_imgs = torch.from_numpy(imgs_padded) / 255

    return ims, num_images, filenames, shape_orig, shape_infer, tensor_imgs


def yolov5_preprocessing(dataset, target_height, target_width):
    data = []
    for image_file_path, label in dataset.data:
        ims, n, files, orig_shape, scaled_shape, pixel_values = data_preprocessing(
            Image.open(image_file_path), size=(target_height, target_width)
        )
        # YOLOv5 needs original shape and scaled shape to scale boxes
        full_label = {
            "image_id": label["image_id"],
            "orig_shape": orig_shape[0],
            "scaled_shape": scaled_shape,
        }
        inputs = pixel_values.squeeze()
        data.append((inputs, full_label))

    return data


def data_postprocessing(
    ims: list,
    x_shape: torch.Size,
    pred: list,
    model: yolov5.models.common.AutoShape,
    n: int,
    shape0: list,
    shape1: list,
    files: list,
) -> Detections:
    """Data postprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : list
        List of input images
    x_shape : torch.Size
        Shape of each image
    pred : list
        List of model predictions
    model : yolov5.models.common.AutoShape
        Model
    n : int
        Number of input samples
    shape0 : list
        Image shape
    shape1 : list
        Inference shape
    files : list
        Filenames

    Returns
    -------
    Detections
        Detection object
    """

    # Create dummy dt tuple (not used but required for Detections)
    dt = (Profile(), Profile(), Profile())

    # Perform NMS
    y = non_max_suppression(
        prediction=pred,
        conf_thres=model.conf,
        iou_thres=model.iou,
        classes=None,
        agnostic=model.agnostic,
        multi_label=model.multi_label,
        labels=(),
        max_det=model.max_det,
    )

    # Scale bounding boxes
    for i in range(n):
        scale_boxes(shape1, y[i][:, :4], shape0[i])

    # Return Detections object
    return Detections(ims, y, files, times=dt, names=model.names, shape=x_shape)


def coco_res_format(nms_preds, batch_labels, rescale=True):
    coco_res_formatted = []
    for preds, labs in zip(nms_preds, batch_labels):
        bboxs = preds[:, :4].clone()
        if rescale:
            # scale bounding boxes to original image shape, need in xyxy format
            _ = scale_boxes(labs["scaled_shape"], bboxs, labs["orig_shape"])
        # convert bbox to COCO format [x, y, width, height]
        bboxs = xyxy2xywh(bboxs)
        # convert xy center to top-left corner
        bboxs[:, :2] -= bboxs[:, 2:] / 2
        # NOTE: YOLOv5 has a different class label mapping than COCO
        res = [
            {
                "image_id": labs["image_id"],
                "category_id": coco80_to_coco91_class()[int(cat.item())],
                "bbox": bbox.tolist(),
                "score": score.item(),  # Confidence score for the prediction
            }
            for bbox, score, cat in zip(bboxs, preds[:, 4], preds[:, 5])
        ]
        coco_res_formatted.extend(res)

    return coco_res_formatted


def coco_post_process_bbox(
    outputs, labels, conf_thres=0.001, iou_thres=0.60, agnostic=False, multi_label=True, max_det=100, rescale=True
):
    # convert predictions to COCO eval format
    # for parameter information see: https://github.com/ultralytics/yolov5/issues/1466
    # iterate over batches
    coco_res = []
    for bouts, blabs in zip(outputs, labels):
        pred_raw = bouts[0].to_pytorch() if not isinstance(bouts[0], torch.Tensor) else bouts
        # get bbox predictions, reject overlapping detections
        nms_preds = non_max_suppression(
            prediction=pred_raw,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=None,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=(),
            max_det=max_det,
        )
        coco_res.extend(coco_res_format(nms_preds, blabs, rescale=rescale))

    return coco_res


def run_coco_eval(coco_res, image_ids, ann_type, data_dir, split="val2017"):
    # run COCO evaluation
    # must select only the image_ids that are predicted
    # otherwise evaluation will give incorrect for all uninferred images
    assert ann_type in ["bbox", "segmentation", "keypoints"]
    ann_prefix = "person_keypoints" if ann_type == "keypoints" else "instances"
    ann_file = f"{data_dir}/annotations/{ann_prefix}_{split}.json"
    coco_true = COCO(ann_file)
    coco_pred = coco_true.loadRes(coco_res)
    coco_eval = COCOeval(coco_true, coco_pred, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # mean average precision at 50%:95% IoU
    m_ap = coco_eval.stats[0]
    return m_ap
