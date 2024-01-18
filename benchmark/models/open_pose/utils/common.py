import math

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
Preprocessing functions for OpenPose model.
https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/datasets/coco_hpe2_dataset.py
"""


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_image(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    top = int(math.floor((min_dims[0] - h) / 2.0))
    left = int(math.floor((min_dims[1] - w) / 2.0))
    bottom = int(min_dims[0] - h - top)
    right = int(min_dims[1] - w - left)
    pad = [top, left, bottom, right]
    padded_img = cv2.copyMakeBorder(
        src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img, pad


def preprocess(image, image_id, target_height, target_width):
    img_mean = (128, 128, 128)
    img_scale = 1.0 / 256
    stride = 8
    pad_value = (0, 0, 0)

    height, width, _ = image.shape
    image = normalize(image, img_mean, img_scale)
    # scale image to so greater dimension is equal to base_height
    ratio = min(target_height / float(height), target_width / float(width))
    image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    # pad image to ensure output size is base_height x base_width
    min_dims = [target_height, target_width]
    image, pad = pad_image(image, stride, pad_value, min_dims)
    image = image.astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)

    label = np.array([image_id, 1.0] + pad + [height, width], np.float32)
    label = torch.from_numpy(label)

    return image, label


def openpose_preprocess(dataset, target_height, target_width):
    data = []
    for image_file_path, label in dataset.data:
        image = cv2.imread(image_file_path, flags=cv2.IMREAD_COLOR)
        image_id = label["image_id"]
        image_data, label = preprocess(image, image_id, target_height, target_width)
        data.append((image_data, label))

    return data


# ---------------------------------------------------------------------------------------------------------------------

# added coco evaluation functions
def to_coco_json_format_batch(b_coco_keypoints, b_scores, b_image_ids):
    # always use category_id=1 for person
    coco_result = [
        {"image_id": img_id, "category_id": 1, "keypoints": coco_kp, "score": scr}
        for coco_kps, scrores, img_ids in zip(b_coco_keypoints, b_scores, b_image_ids)
        for coco_kp, scr, img_id in zip(coco_kps, scrores, img_ids)
    ]
    return coco_result


def to_coco_json_format(coco_keypoints, scores, image_ids):
    # always use category_id=1 for person
    coco_result = [
        {"image_id": img_id, "category_id": 1, "keypoints": coco_kp, "score": scr}
        for coco_kp, scr, img_id in zip(coco_keypoints, scores, image_ids)
    ]
    return coco_result


def run_coco_eval(coco_res, image_ids, ann_type, data_dir, split="val2017"):
    # run COCO evaluation
    # must select only the image_ids that are in the coco_res
    # otherwise evaluation will give incorrect for all uninferred images
    assert ann_type in ["bbox", "segmentation", "keypoints"]
    ann_prefix = "person_keypoints" if ann_type == "keypoints" else "instances"
    ann_file = f"{data_dir}/annotations/{ann_prefix}_{split}.json"
    coco_true = COCO(ann_file)
    if not coco_res:
        # empty list of results fails evaluation test
        return 0.0
    coco_pred = coco_true.loadRes(coco_res)
    coco_eval = COCOeval(coco_true, coco_pred, ann_type)
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # mean average precision:
    # bbox: 50%:95% IoU
    # keypoints: OKS
    m_ap = coco_eval.stats[0]
    return m_ap
