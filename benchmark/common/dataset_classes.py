# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from skimage.io import imread
from torch.utils.data import Dataset
from tqdm import tqdm


class DummyNLPDataset(Dataset):
    """Dummy dataset class used for generating random data for NLP models."""

    def __init__(self, microbatch: int, seq_len: int, hidden_size: int, type_ids: str = None):
        """Init function to generate random dataset.

        Parameters
        ----------
        microbatch : int
            Batch size
        seq_len : int
            Sequence length
        hidden_size : int
            Hidden size of latent space
        type_ids : str
            Name of variable input parameter
        """

        if type_ids:
            self.data = [
                (
                    {
                        "input_ids": torch.randint(high=25000, size=(seq_len,)),
                        "attention_mask": torch.randint(high=2, size=(seq_len,)),
                        type_ids: torch.randint(high=2, size=(seq_len,)),
                    },
                    torch.rand(seq_len, hidden_size),
                )
                for _ in range(microbatch)
            ]
        else:
            self.data = [
                (
                    {
                        "input_ids": torch.randint(high=25000, size=(seq_len,)),
                        "attention_mask": torch.randint(high=2, size=(seq_len,)),
                    },
                    torch.rand(seq_len, hidden_size),
                )
                for _ in range(microbatch)
            ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class DummyPipelineDataset(Dataset):
    """Dummy dataset class used for generating random data for pipeline models."""

    def __init__(self, microbatch: int, sample_text: str, answer: str):
        """Init function to generate random dataset.

        Parameters
        ----------
        microbatch : int
            Batch size
        sample_text : str
            Sample text
        """

        self.data = [(sample_text, answer) for _ in range(microbatch)]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class DummyCVDataset(Dataset):
    """Dummy dataset class used for generating random data for CV models."""

    def __init__(self, microbatch: int, channels: int, height: int, width: int, data_type: str):
        """Init function to generate random dataset.

        Parameters
        ----------
        microbatch : int
            Batch size
        channels : int
            Number of colour channels
        height : int
            Image height resolution
        width : int
            Image width resolution
        """

        self.data = [
            (
                [torch.rand(size=(channels, height, width))],
                torch.rand(1),
            )
            for _ in range(microbatch)
        ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class DummyCVDataset_1D(Dataset):
    """Dummy dataset class used for generating random data for CV models."""

    def __init__(self, microbatch: int, channels: int, height: int, width: int, data_type: str):
        """Init function to generate random dataset.

        Parameters
        ----------
        microbatch : int
            Batch size
        channels : int
            Number of colour channels
        height : int
            Image height resolution
        width : int
            Image width resolution
        """

        self.data = [
            (
                [torch.rand(size=(channels, int(height * width)))],
                torch.rand(1),
            )
            for _ in range(microbatch)
        ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class DummyAudioDataset(Dataset):
    """Dummy dataset class used for generating random data for Audio models."""

    def __init__(
        self,
        microbatch: int,
        seq_len: int,
        feature_size: int,
        hidden_size: int,
        type_ids: str,
        data_type: torch.dtype = None,
    ):
        """Init function to generate random dataset.

        Parameters
        ----------
        microbatch : int
            Batch size
        seq_len : int
            Sequence length
        hidden_size : int
            Hidden size of latent space
        type_ids : str
            Name of variable input parameter
        """

        self.data = [
            (
                {
                    "input_features": torch.rand(size=feature_size, dtype=data_type),
                    type_ids: torch.randint(high=2, size=(seq_len,)),
                },
                torch.rand(seq_len, hidden_size),
            )
            for _ in range(microbatch)
        ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class SST2Dataset(Dataset):
    """Configurable SST-2 Dataset."""

    def __init__(self, dataset: Any, tokenizer: Any, split: str, seq_len: int):
        """Init and preprocess SST-2 dataset.

        Parameters
        ----------
        dataset : Any
            SST-2 dataset
        tokenizer : Any
            tokenizer object from HuggingFace
        split : str
            Which split to use i.e. ["train", "validation", "test"]
        seq_len : int
            Sequence length
        """
        self.sst2 = dataset[split]
        self.data = [
            (
                tokenizer(
                    item["sentence"],
                    return_tensors="pt",
                    max_length=seq_len,
                    padding="max_length",
                    return_token_type_ids=False,
                    truncation=True,
                ),
                item["label"],
            )
            for item in self.sst2
        ]

        for data in self.data:
            tokenized = data[0]
            for item in tokenized:
                tokenized[item] = tokenized[item].squeeze()

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class PipelineDataset(Dataset):
    """Dataset class used for generating data for pipeline models."""

    def __init__(
        self,
        dataset: Any,
        input_text: str,
        label: str,
        prepend_text: str = "",
        use_feature: bool = False,
        serialize_hellaswag: bool = False,
        tokenizer: Any = None,
    ):
        """Init function to generate random dataset.

        Parameters
        ----------
        dataset : Any
            Dataset
        input_text : str
            Dictionary key for input data
        label : str
            Dictionary key for label data
        prepend_text : str
            Text to prepend input with i.e. "sst2 sentence: "
        use_feature : bool
            Whether to use dataset feature as a prepend or not
        serialize_hellaswag : bool
            Whether to serialize the hellaswag dataset or not
        tokenizer : Any
            tokenizer object from HuggingFace
        """
        if use_feature:
            self.data = [(item[prepend_text] + ": " + item[input_text], item[label]) for item in dataset]
        elif serialize_hellaswag:
            self.data = []
            for item in dataset:
                serialized = [
                    (
                        f"{item[prepend_text]}: {item[input_text]} {end}",
                        {
                            "ind": item["ind"],
                            "label": item[label],
                            "end_ids": tokenizer(end, padding=True, return_tensors="pt").input_ids.squeeze(),
                            "prompt_ids": tokenizer(
                                f"{item[prepend_text]}: {item[input_text]}", padding=True, return_tensors="pt"
                            ).input_ids.squeeze(),
                        },
                    )
                    for end in item["endings"]
                ]
                self.data.extend(serialized)
        else:
            self.data = [(prepend_text + item[input_text], item[label]) for item in dataset]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class StackExchangeDataset(Dataset):
    """Configurable StackExchange Dataset."""

    def __init__(self, dataset: Any, tokenizer: Any, split: str, seq_len: int):
        """Init and preprocess memray dataset.

        Parameters
        ----------
        dataset : Any
            stackexchange dataset: memray/stackexchange
        tokenizer : Any
            tokenizer object from HuggingFace
        split : str
            Which split to use i.e. ["train", "validation", "test"]
        seq_len : int
            Sequence length
        """
        if split:
            self.stackexchange = dataset[split]
        else:
            self.stackexchange = dataset

        self.data = []

        # note: labels are expressed as masks into the input token sequence, that is the output of the model
        for item in self.stackexchange:
            labels = tokenizer(
                item["tags"].replace(";", " "),
                return_tensors="pt",
                max_length=seq_len,
                padding="max_length",
                return_token_type_ids=False,
                truncation=True,
            )["input_ids"]
            # remove special tokens
            labels = labels[~torch.isin(labels, torch.tensor([0, 101, 102]))]
            tokens = tokenizer(
                item["question"],
                return_tensors="pt",
                max_length=seq_len,
                padding="max_length",
                return_token_type_ids=False,
                truncation=True,
            )
            label_mask = torch.isin(tokens["input_ids"], labels)
            self.data.append((tokens, label_mask))

        for data in self.data:
            tokenized = data[0]
            for item in tokenized:
                tokenized[item] = tokenized[item].squeeze()

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class TweetEval(Dataset):
    """Configurable TweetEval Dataset."""

    def __init__(self, dataset: Any, tokenizer: Any, split: str, seq_len: int):
        """Init and preprocess TweetEval dataset.

        Parameters
        ----------
        dataset : Any
            TweetEval dataset
        tokenizer : Any
            tokenizer object from HuggingFace
        split : str
            Which split to use i.e. ["train", "validation", "test"]
        seq_len : int
            Sequence length
        """
        self.tweeteval = dataset[split]
        self.data = [
            (
                tokenizer(
                    self._preprocess(item["text"]),
                    return_tensors="pt",
                    max_length=seq_len,
                    padding="max_length",
                    truncation=True,
                ),
                item["label"],
            )
            for item in self.tweeteval
        ]

        for data in self.data:
            tokenized = data[0]
            for item in tokenized:
                tokenized[item] = tokenized[item].squeeze()

    def _preprocess(self, text: str):
        """Preprocess function for Tweets

        Parameters
        ----------
        text : str
            Original text

        Returns
        -------
        str
            Preprocessed text
        """
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)

        return " ".join(new_text)

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class ImageNetDataset(Dataset):
    """Configurable ImageNet Dataset."""

    def __init__(self, dataset: Any, feature_extractor: Any, version=None):
        """Init and preprocess ImageNet-1k dataset.

        Parameters
        ----------
        dataset : Any
            ImageNet-1k dataset
        feature_extractor : Any
            feature extractor object from HuggingFace
        """
        self.imagenet = dataset

        # for MobileNetV3+Timm or vovNet v2
        if version in ["timm", "torch"]:
            self.data = [
                (
                    [feature_extractor(item["image"].convert("RGB"))],
                    item["label"],
                )
                for item in self.imagenet
            ]

        # for huggingface models
        else:
            self.data = [
                (
                    [feature_extractor(item["image"].convert("RGB"), return_tensors="pt")["pixel_values"].squeeze()],
                    item["label"],
                )
                for item in self.imagenet
            ]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class LibriSpeechDataset(Dataset):
    """
    URL: https://huggingface.co/datasets/librispeech_asr
    sample rate: 16 kHz

    CC 4.0 license: https://creativecommons.org/licenses/by/4.0/
    """

    def __init__(self, dataset: Any):
        """Init function to generate random dataset.

        Parameters
        ----------
        dataset : Any
            LibriSpeech dataset
        """
        self.data = [(sample["audio"]["array"], sample["text"]) for sample in dataset]

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class COCODataset(Dataset):
    """Configurable Common Objects in Context (COCO) Dataset.
    License:
        annotations CC 4.0 https://cocodataset.org/#termsofuse
        images: CC https://www.flickr.com/creativecommons/
        URL: (https://cocodataset.org/)

    # requires the `COCO API` to be installed
    # Note: cannot use `torchvision.datasets.CocoDetection` because it does
    # preprocessing during __getitem__ which adds overhead to benchmarking
    # adapted from torchvision.datasets.CocoDetection
    # see: https://github.com/pytorch/vision/blob/main/torchvision/datasets/coco.py
    """

    def __init__(self, data_dir: str, split: str, ann_type: str, n_samples: int):
        """Init and preprocess COCO dataset

        Parameters
        ----------
        data_dir : str
            Directory where dataset is stored, e.g. "${MLDATA_DIR}/coco"
        split : str
            Dataset split to use, e.g. "val2017"
        ann_type : str
            Type of annotations to use, e.g. "bbox", in ["bbox", "segmentation", "keypoints"]
        n_samples : int
            Number of samples to use from dataset
        """
        from pycocotools.coco import COCO

        assert ann_type in ["bbox", "segmentation", "keypoints"]
        assert split in ["val2017"]  # only downloaded val2017 dataset
        ann_prefix = "person_keypoints" if ann_type == "keypoints" else "instances"
        data_dir = Path(data_dir)
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=False)
            self.download_dataset(data_dir, split)

        ann_file = f"{data_dir}/annotations/{ann_prefix}_{split}.json"
        image_dir = Path(data_dir) / "images" / split
        # initialize COCO api for instance annotations
        coco = COCO(ann_file)

        self.data = []
        for n, (img_id, coco_img) in enumerate(coco.imgs.items()):
            if n >= n_samples:
                break
            fpath = image_dir / coco_img["file_name"]
            self.data.append([fpath, {"image_id": img_id}])

    def download_dataset(self, data_dir, split):
        data_dir = Path(data_dir)
        print(f"downloading COCO {split} dataset to: {data_dir}")
        url_split_map = {
            "val2017": {
                "images": "http://images.cocodataset.org/zips/val2017.zip",
                "": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            }
        }
        url_dict = url_split_map[split]
        for dest_name, url in url_dict.items():
            temp_zip = data_dir / "_temp.zip"
            dest = data_dir / dest_name

            res = requests.get(url, stream=True)

            # Get the total file size
            file_size = int(res.headers.get("content-length", 0))

            # Create a progress bar using tqdm
            progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)
            with open(temp_zip, "wb") as f:
                for chunk in res.iter_content(chunk_size=1024):
                    if chunk:
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            progress_bar.close()

            with zipfile.ZipFile(temp_zip, "r") as zip_ref:
                zip_ref.extractall(dest)

            # delete temporary zip file
            temp_zip.unlink()

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y


class BrainSegmentationDataset(Dataset):
    """
    Brain MRI dataset for FLAIR abnormality segmentation.

    uses lgg_segmentation dataset (Low Grade Glioma Segmentation)
    url: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

    code adapted from: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/dataset.py
    """

    def __init__(
        self,
        data_dir,
        image_size=256,
        subset="validation",
        n_samples=10,
    ):
        from benchmark.models.unet.brain_segmentation.utils import (
            crop_sample,
            normalize_volume,
            pad_sample,
            resize_sample,
        )

        data_dir = Path(data_dir)
        if not data_dir.exists():
            self.download_dataset(data_dir)

        images_dir = Path(data_dir) / "kaggle_3m"

        # read images
        volumes = {}
        masks = {}
        count_samples = 0
        self.patients = []
        # NOTE: do not use os.walk, does not provide the same ordering on all systems
        for dirpath in sorted([d for d in images_dir.iterdir() if d.is_dir()]):
            filenames = [f for f in dirpath.iterdir() if f.is_file()]
            image_slices = []
            mask_slices = []
            if count_samples > n_samples:
                break
            for filename in sorted(
                filenames,
                key=lambda x: int(x.name.split(".")[-2].split("_")[4]),
            ):
                if "mask" in filename.name:
                    mask_slices.append(imread(filename, as_gray=True))
                else:
                    image_slices.append(imread(filename))
            if len(image_slices) > 0:
                patient_id = dirpath.name
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])
                count_samples += len(volumes[patient_id])
                self.patients.append(patient_id)

        # NOTE: sorting self.patients varies which idx corresponds to
        # to which patient when n_samples changes, do not sort

        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        # finally get data per image not per patient
        self.data = []
        for (patient_id, slice_id) in self.patient_slice_index[:n_samples]:
            v, m = self.volumes[patient_id]
            # fix dimensions (C, H, W)
            image = v[slice_id].transpose(2, 0, 1)
            label_mask = m[slice_id].transpose(2, 0, 1)
            image_tensor = torch.from_numpy(image.astype(np.float32))
            label_mask_tensor = torch.from_numpy(label_mask.astype(np.float32))
            self.data.append([image_tensor, label_mask_tensor])

        # clean up
        del self.volumes, self.patient_slice_index

    def download_dataset(self, data_dir):
        print("You must download dataset.")
        print("Kaggle download requires making an account and authenticating.")
        print("url: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation")
        print(f"steps: \n1. download dataset from url \n2. unzip in: {data_dir}")

    def __len__(self):
        """Return length of dataset.

        Returns
        -------
        int
            Length of dataset
        """
        return len(self.data)

    def __getitem__(self, index: int):
        """Return sample from dataset.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        Tuple
            Data sample in format of X, y
        """
        X, y = self.data[index]
        return X, y
