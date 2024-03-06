# Benchmarking

Repository for AI model benchmarking on accelerator hardware.

## Setup Instructions

### Installation

First, create either a Python virtual environment with PyBuda installed or execute from a Docker container with PyBuda installed.

Installation instructions can be found at [Install TT-BUDA](https://github.com/tenstorrent/tt-buda-demos/blob/main/first_5_steps/1_install_tt_buda.md).

Next, install the model requirements:

```bash
pip install -r requirements.txt
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

To use some datasets for evaluation benchmarking, such as ImageNet-1k, you will need to be connected to HuggingFace Hub.

1. Create a HuggingFace account from <https://huggingface.co/>
2. Create a User Access Token by following steps in <https://huggingface.co/docs/hub/security-tokens>
3. Download the `huggingface_hub` library and login through `huggingface-cli` with your User Access Token by following the steps in <https://huggingface.co/docs/huggingface_hub/quick-start>

#### Benchmarking Datasets Setup

HuggingFace datasets will download into `${HF_HOME}/datasets/` once HuggingFace Hub access is setup.

COCO: is downloaded automatically from <https://cocodataset.org>, no login is needed. The cache location is `~/.cache/coco`

LGG Segmentation Dataset: must be manually downloaded from <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>, this requires a Kaggle login. The commands below are to extract the downloaded archive into the correct location: `~/.cache/mldata/lgg_segmentation/kaggle_3m`.

```bash
# use another location for MLDATA_DIR if desired, below is default
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
python benchmark.py -d tt -m bert -c tiny --task text_classification --save_output
```

or

```bash
python benchmark.py -d cuda -m bert -c tiny --task text_classification --save_output
```

To see which models and configurations are currently supported, run:

```bash
benchmark.py --list
```

### Run Examples

You can find example commands for various conditions in the file:

- `run_benchmark_tt_perf` for TT and `run_benchmark_cuda` GPU & CPU devices

## Contributing

We are excited to move our development to the public, open-source domain. However, we are not adequately staffed to review contributions in an expedient and manageable time frame at this time. In the meantime, please review the [contributor's guide](CONTRIBUTING.md) for more information about contribution standards.

## Communication

If you would like to formally propose a new feature, report a bug, or have issues with permissions, please file through [GitHub issues](https://github.com/tenstorrent/benchmarking/issues).

Please access the [Discord community](https://discord.gg/xUHw4tMcRV) forum for updates, tips, live troubleshooting support, and more!
