# Benchmarking

Repository for AI model benchmarking on accelerator hardware.

## Performance Table

| Model                                                      | Input Size          | Batch      | Grayskull e75  | Grayskull e150   | Wormhole n150 | Wormhole n300 (single-chip) | Wormhole n300 (dual-chip) | TT-LoudBox/TT-QuietBox (4 MMIO chip) | TT-LoudBox/TT-QuietBox (8 chip) |
|----------------------------------------------------------  |---------------------|----------- |----------------|------------------|---------------|-----------------------------|---------------------------|---------------------------|---------------------------|
| [BERT-Large](./benchmark/models/bert) (sen/s)              | 384                 | 64         | 81             | 99               | 118           | x                           | x                         | x                         | x                         |
| [T5-Large](./benchmark/models/t5) (tok/s)                  | 64                  | 1          | 25             | 30               | 75            | 68                          | x                         | x                         | x                         |
| [FLAN-T5-Large](./benchmark/models/flant5) (tok/s)         | 64                  | 1          | 9              | 25               | 71            | 52                          | x                         | x                         | x                         |
| [Whisper-Small](./benchmark/models/whisper) (tok/s)        | 30s                 | 1          | 3.4            | 3.7              | 16            | 10                          | x                         | x                         | x                         |
| [Falcon-7B](./benchmark/models/falcon) (tok/s)             | 128                 | 32         | x              | x                | 76            | 77                          | x                         | x                         | x                         |
| [SD v1-4](./benchmark/models/stable_diffusion) (s/img)     | 512x512             | 1          | x              | x                | 50            | 50                          | x                         | x                         | x                         |
| [ResNet50](./benchmark/models/resnet) (img/s)              | 3x224x224           | 256        | 1106           | 1410             | 2891          | 1060                        | 2000                      | 3315                         | 4711                         |
| [VoVNet-V2](./benchmark/models/vovnet) (img/s)             | 3x224x224           | 128        | 518            | 819              | 1603          | 1197                        | 1931                      | 2860                         | 3294                         |
| [MobileNetV1](./benchmark/models/mobilenetv1) (img/s)      | 3x224x224           | 128        | 2468           | 2924             | 3102          | 2338                        | 2978                      | 4334                         | 4347                         |
| [MobileNetV2](./benchmark/models/mobilenetv2) (img/s)      | 3x224x224           | 256        | 1141           | 1491             | 2721          | 2439                        | 4332                      | 4579                         | 6800                         |
| [MobileNetV3](./benchmark/models/mobilenetv3) (img/s)      | 3x224x224           | 64         | 1192           | 1741             | 1981          | 1670                        | 2017                      | 2695                         | 1688                         |
| [HRNet-V2](./benchmark/models/hrnet) (img/s)               | 3x224x224           | 128        | 197            | 233              | 324           | 257                         | 269                       | 845                         | 262                         |
| [ViT-Base](./benchmark/models/vit) (img/s)                 | 3x224x224           | 64         | 301            | 363              | 540           | 447                         | 546                       | 970                         | 1311                         |
| [DeiT-Base](./benchmark/models/deit) (img/s)               | 3x224x224           | 64         | 301            | 363              | 539           | 446                         | 545                       | 973                         | 1317                         |
| [YOLOv5-Small](./benchmark/models/yolo_v5) (img/s)         | 3x320x320           | 128        | 290            | 232              | 1190          | 1090                        | 1435                      | x                         | x                         |
| [OpenPose-2D](./benchmark/models/openpose) (img/s)         | 3x224x224           | 64         | 828            | 1098             | 1252          | 1204                        | 1805                      | 1542                         | 1438                         |
| [U-Net](./benchmark/models/unet) (img/s)                   | 3x256x256           | 48         | 222            | 268              | 490           | 344                         | 547                       | 455                         | x                         |
| [Inception-v4](./benchmark/models/inception_v4) (img/s)    | 3x224x224           | 128        | 371            | 458              | 567           | 1116                        | 1810                      | 2795                         | 3162                         |

## Setup Instructions

### Installation

First, create either a Python virtual environment with PyBuda installed or execute from a Docker container with PyBuda installed.

Installation instructions can be found at [Install TT-BUDA](https://github.com/tenstorrent/tt-buda-demos/blob/main/first_5_steps/1_install_tt_buda.md).

Next, install the model requirements:

```bash
pip install -r requirements.txt --constraint constraints.txt
```

### Installing on a System with a GPU

If your system contains a GPU device, you can use the `requirements-cuda.txt` file to install the correct dependencies.

To install packages, make sure your virtual environment is active.

#### To add additional CUDA packages

```bash
pip install -r requirements-cuda.txt
```

### Environment Setup

#### Setup Access to HuggingFace Hub

To access the benchmarking datasets, follow these steps to set up your access to the HuggingFace Hub:

1. **Create a HuggingFace Account**:
   - Visit [Hugging Face](https://huggingface.co/) and create an account if you haven't already.

2. **Generate User Access Token**:
   - Follow the steps outlined in the [HuggingFace Docs - Security Tokens](https://huggingface.co/docs/hub/security-tokens) to generate a User Access Token.

3. **Install `huggingface_hub` Library**:
   - Install the `huggingface_hub` library by running:

     ```bash
     pip install huggingface_hub
     ```

4. **Login to HuggingFace CLI**:
   - Login to the HuggingFace CLI using your User Access Token:

     ```bash
     huggingface-cli login
     ```

   - Enter your User Access Token when prompted.

5. **Validate Setup**:
   - Run the following command to verify your login status:

     ```bash
     huggingface-cli whoami
     ```

   - If your username is displayed, it means you are successfully logged in.

6. **Dataset Access**:
   - Visit [HuggingFace Datasets - ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) and follow the instructions to grant access to the ImageNet-1k dataset.

## Validation Steps

After completing the setup process, ensure that everything is working correctly:

1. **Verify Hugging Face Hub Login**:
   - Run the following command to verify that you are logged in to the Hugging Face Hub:

     ```bash
     huggingface-cli whoami
     ```

   - If your username is displayed, it means you are successfully logged in.

2. **Check Dataset Access**:
   - Visit the [HuggingFace Datasets - ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) page.
   - Make sure you can view and access the dataset details without any permission errors.

3. **Accept Dataset Access (If Required)**:
   - If you encounter any permission errors while accessing the ImageNet-1k dataset, ensure that you follow the instructions provided on the dataset page to grant access.

### Benchmarking Datasets Setup

To set up the three required datasets for running benchmarking tests within this repository, follow these steps for each dataset:

1. **HuggingFace datasets**: will download into `${HF_HOME}/datasets/` once [HuggingFace Hub access](#setup-access-to-huggingface-hub) is set up.

2. **COCO Dataset**: You can automatically download the COCO dataset from [here](https://cocodataset.org/#download:~:text=2017%20Val%20images%20%5B5K/1GB%5D). No login is required, and the dataset will be cached in `~/.cache/coco`.

   To download the COCO dataset, follow these steps:

   ```bash
   # use another location for MLDATA_DIR if desired, below is default
   # Create the `coco` directory inside the cache directory:
   mkdir -p ~/.cache/mldata/coco

   # Navigate to the `coco` directory:
   cd ~/.cache/mldata/coco

   # Create the `images` directory:
   mkdir images
   cd images

   # Download the COCO validation images:
   wget http://images.cocodataset.org/zips/val2017.zip

   # Unzip the downloaded file:
   unzip val2017.zip

   # Move back to the `coco` directory:
   cd ..

   # Download the COCO train/val annotations:
   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

   # Unzip the downloaded file:
   unzip annotations_trainval2017.zip
   ```

3. **LGG Segmentation Dataset**: must be manually downloaded from <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>, this requires a Kaggle login. The commands below are to extract the downloaded archive into the correct location: `~/.cache/mldata/lgg_segmentation/kaggle_3m`.

   ```bash
   # use another location for MLDATA_DIR if desired, below is default;
   # Download and move the downloaded archive and unzip within the lgg_segmentation folder.
   mkdir -p ~/.cache/mldata/lgg_segmentation
   cd ~/.cache/mldata/lgg_segmentation
   # download and move the archive here then unzip
   unzip archive.zip
   # the dataset appears to have two copies that are equivalent, remove the extra one
   rm -r lgg-mri-segmentation
   ```

## Run Benchmarking

As part of the environment setup, you may need to add the root to `PYTHONPATH`:

```bash
export PYTHONPATH=.
```

### Benchmarking Script

`benchmark.py` allows easy way to benchmark performance of a support model, while varying configurations and compiler options. The script will measure
real end-to-end time on host, starting post-compile at first input pushed, and ending at last output received.

To specify the device to run on ("tt", "cpu", or "cuda"), include the `-d` argument flag.

The script optionally outputs a .json file with benchmark results and options used to run the test. If the file already exists, the results will be appended,
allowing the user to run multiple benchmark back-to-back, like:

```bash
python benchmark.py -d tt -m bert -c base --task text_classification --save_output
```

or

```bash
python benchmark.py -d cuda -m bert -c base --task text_classification --save_output
```

To see which models and configurations are currently supported, run:

```bash
benchmark.py --list
```

### Run Examples

You can find example commands for various conditions in the file:

- `run_benchmark_tt_perf` for TT and `run_benchmark_cuda` GPU & CPU devices

## Running On Multi-Device

To run the benchmarks on multichip or multicard systems,

- n300 (multichip)
  - Set env variable `PYBUDA_N300_DATA_PARALLEL=1`
- TT-LoudBox / TT-QuietBox (multicard)
  - set env variable `PYBUDA_FORCE_THREADS=1` and set config `--parallel_tti device_images/`

## Contributing

We are excited to move our development to the public, open-source domain. However, we are not adequately staffed to review contributions in an expedient and manageable time frame at this time. In the meantime, please review the [contributor's guide](CONTRIBUTING.md) for more information about contribution standards.

## Communication

If you would like to formally propose a new feature, report a bug, or have issues with permissions, please file through [GitHub issues](https://github.com/tenstorrent/benchmarking/issues).

Please access the [Discord community](https://discord.gg/xUHw4tMcRV) forum for updates, tips, live troubleshooting support, and more!
