# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import socket
import subprocess
import time
import uuid
from datetime import datetime
from enum import Enum, auto
from importlib.metadata import metadata

import psutil
import torch
import transformers
from diffusers import StableDiffusionPipeline
from transformers.pipelines import Pipeline

from benchmark.common import get_num_tokens_generated

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeviceType(Enum):
    TT = auto()
    CUDA = auto()
    CPU = auto()


class OutputType(Enum):
    TEXT = auto()
    TOKEN_IDS = auto()
    LOGITS = auto()


def get_device_type_from_str(device_str: str) -> DeviceType:
    device_type = None
    if device_str == "tt":
        device_type = DeviceType.TT
    elif device_str == "cuda":
        device_type = DeviceType.CUDA
    elif device_str == "cpu":
        device_type = DeviceType.CPU
    else:
        raise RuntimeError("Unknown device type: " + device_str)
    return device_type


def check_library_importable(library_name):
    try:
        __import__(library_name)
        logging.info(f" {library_name} is importable.")
        return True
    except ImportError:
        logging.warning(f" {library_name} cannot be imported.")
        return False


class BenchmarkRun:
    """
    A class used to represent a benchmarking run of a given ML model.

    A BenchmarkRun object contains metadata handling code
    to determine:
        - which runtime methods should be used for a given benchmark run.
        - which output data and statistics to collect.

    A BenchmarkRun object is initialized as default and can be optionally
    specialized within any model implementation. For example in stable_diffusion.py.
    """

    def __init__(self, args):
        self.run_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H_%M_%S")
        self.run_id = uuid.uuid4()
        self.short_run_id = str(self.run_id)[:5]
        self.run_name = (
            f"{args.model}_{args.device}_{args.config}_mb{args.microbatch}_{self.run_timestamp}_{self.short_run_id}"
        )
        self.load_tti = args.load_tti
        self.save_tti = args.save_tti
        self.loop_count = args.loop_count
        self.microbatch = args.microbatch
        self.dataformat = args.dataformat
        self.chips = args.chips
        self.device_type = get_device_type_from_str(args.device)
        self.device_name = None
        self.device_list = None
        self.arch = None
        self.pybuda_hash = None
        self.pybuda_hash_date = None
        self.torch_version = None
        self.machine_name = None
        # model metadata
        self.model_name = None
        self.model_type = None
        self.input_type = None
        self.output_type = None
        self.input_shape = None
        self.has_compile_inputs = False
        self.has_forward_wrapper = False
        self.eval_post_process = None
        self.store_post_process = None
        self.store_output_func = None
        # benchmark stats
        self.expected_total_samples = None
        self.total_samples = None
        self.total_tokens = None
        self.perf_unit_str = None
        self.perf_value = None
        self.tokenizer = None
        self.peak_cpu_mem = 0
        # benchmark stats metadata
        self.count_prompt_tokens = False
        self.token_count_use_max_batch = False
        # timers
        self.compilation_start_time = None
        self.compilation_end_time = None
        self.compilation_duration = 0
        self.benchmark_start_time = None
        self.benchmark_end_time = None
        self.benchmark_duration = None
        self.stop_monitoring = False
        # initialize with environment metadata
        self.get_pybuda_metadata()
        self.get_device_metadata()

    def default_model_init(self, model_func, model, generator, eval_fn):
        # this function is run by the benchmark decorator wrapper
        self.model_name = model_func.__name__
        self.total_samples = self.loop_count * self.microbatch * len(generator)
        # get model impl metadata
        if self.device_type == DeviceType.TT:
            if isinstance(model, dict):
                if "compile_inputs" in model.keys():
                    self.has_compile_inputs = True
                if "forward_wrapper" in model.keys():
                    self.has_forward_wrapper = True

        self.model_type = type(model)

    def get_device_metadata(self):
        # Get host info
        all_info = subprocess.check_output("cat /proc/cpuinfo", shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                self.host_device_name = re.sub(".*model name.*:", "", line, 1).lstrip()
        # Get device info
        if self.device_type == DeviceType.TT:
            import pybuda

            self.device_list = pybuda.detect_available_devices()
            self.arch = self.device_list[0]
            self.device_name = str(self.arch).split(".", 1)[1]
        elif self.device_type == DeviceType.CUDA:
            import nvidia_smi

            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self.device_name = nvidia_smi.nvmlDeviceGetName(handle).decode()
        else:
            self.device_name = self.host_device_name
        self.torch_version = torch.__version__
        self.machine_name = socket.gethostname()

    def get_pybuda_metadata(self):
        if not check_library_importable("pybuda"):
            return
        # Get PyBUDA information
        pybuda_pkg_meta = metadata("pybuda")
        if pybuda_pkg_meta.get("Author") and pybuda_pkg_meta.get("Author") != "UNKNOWN":
            # the Author field is blank for source dist installs
            assert (
                pybuda_pkg_meta.get("Author") == "Tenstorrent"
            ), f"Author:={pybuda_pkg_meta.get('Author')}, is this a valid wheel dist?"
            self.pybuda_hash = pybuda_pkg_meta.get("Version")
            self.pybuda_hash_date = ""
        else:
            logger.info(" PyBUDA is not installed as a wheel dist, get git hash from submodule.")
            self.pybuda_hash = (
                subprocess.check_output(["git", "submodule", "status"]).decode("ascii").strip().replace("+", "")[:8]
            )
            current_working_directory = os.getcwd()
            pybuda_dir = current_working_directory + "/external_libs/pybuda/"
            if os.path.exists(pybuda_dir):
                os.chdir(pybuda_dir)
                self.pybuda_hash_date = (
                    subprocess.check_output(["git", "log", "-1", "--format=%cd"]).decode("ascii").strip()
                )
            else:
                self.pybuda_hash_date = "n/a"
            os.chdir(current_working_directory)

    def set_input_shape(self, input_data, model):
        # input_data has the uniform dimensions of:
        # [[num_batch], [input, target], [batch_size], [input_shape]]]
        inputs = input_data[0][0]
        if isinstance(model, Pipeline) or isinstance(model, StableDiffusionPipeline) or self.has_compile_inputs:
            input_shape = "NA"
        elif isinstance(inputs, dict) or isinstance(inputs, transformers.tokenization_utils_base.BatchEncoding):
            input_shape = list(inputs.values())[0].shape[1:]
        elif hasattr(inputs[0], "shape"):
            input_shape = inputs[0].shape[1:]
        else:
            input_shape = "NA"

        self.input_shape = input_shape
        logger.info(f" Input shape: {self.input_shape}")

    def get_shape_str(self, shape):
        if shape == "NA" or shape is None:
            shape_str = "NA"
        else:
            shape_str = "x".join(str(dim) for dim in shape)
        return shape_str

    def get_input_shape_str(self):
        return self.get_shape_str(self.input_shape)

    def start_compilation_timer(self):
        self.compilation_start_time = time.time()
        logger.info(f" Starting compilation at: {time.asctime(time.localtime(self.compilation_start_time))}")

    def end_compilation_timer(self):
        self.compilation_end_time = time.time()
        self.compilation_duration = self.compilation_end_time - self.compilation_start_time
        logger.info(f" Ending compilation at: {time.asctime(time.localtime(self.compilation_end_time))}")
        logger.info(f" Compilation duration: {self.compilation_duration} seconds")

    def start_benchmark_timer(self):
        self.benchmark_start_time = time.time()
        print("*****************************************************")
        print(f" Starting benchmark at: {time.asctime(time.localtime(self.benchmark_start_time))}.")
        print("*****************************************************")

    def end_benchmark_timer(self):
        self.benchmark_end_time = time.time()
        self.benchmark_duration = self.benchmark_end_time - self.benchmark_start_time
        print("*****************************************************")
        print(" Ending benchmark at ", time.asctime(time.localtime(self.benchmark_end_time)))
        print("*****************************************************")

    def cpu_usage_monitor(self):
        baseline_cpu_mem_usage = psutil.virtual_memory().used
        while not self.stop_monitoring:
            cpu_mem_usage = (psutil.virtual_memory().used - baseline_cpu_mem_usage) / (1000**2)
            self.peak_cpu_mem = cpu_mem_usage if cpu_mem_usage > self.peak_cpu_mem else self.peak_cpu_mem
            time.sleep(1)

    def calc_output_stats(self, output, model, eval_score):
        self.eval_score = eval_score
        # get number of tokens generated and samples outputs produced
        tokenizer = None
        output_key = None
        if self.has_forward_wrapper and (isinstance(model, dict)):
            tokenizer = model.get("tokenizer")
            output_key = model.get("output_key")
        elif (
            (isinstance(model, Pipeline) and not isinstance(model, StableDiffusionPipeline))
            and hasattr(model, "tokenizer")
            and hasattr(model, "output_key")
        ):
            tokenizer = model.tokenizer
            output_key = model.output_key

        self.total_tokens = 0
        if tokenizer is not None and output_key is not None:
            # outputkey is used for pipeline implementations that need to be tokenized
            # to determine the number of tokens generated
            num_batch_tokens = [
                sum(get_num_tokens_generated(b_out, model.tokenizer, model.output_key)) for b_out in output
            ]
            self.total_tokens = sum(num_batch_tokens)
            # TODO: add checksum with expected number of tokens if given.
        elif self.has_forward_wrapper and tokenizer is not None and output_key is None:
            self.total_tokens = sum([len(out) for b_out in output for out in b_out])
        elif (self.output_type == OutputType.TEXT and self.tokenizer is not None) or (
            self.output_type == OutputType.LOGITS
        ):
            _total_tokens = 0
            for b_out in output:
                if self.tokenizer is not None and self.output_type == OutputType.TEXT:
                    # only tokenize to get number of tokens if output is text
                    batch_tokens = [len(self.tokenizer(out).input_ids) for out in b_out]
                else:
                    batch_tokens = [len(out) for out in b_out]
                if self.token_count_use_max_batch:
                    # for TT devices that generate for all user_rows in a batch
                    _total_tokens += max(batch_tokens) * len(batch_tokens)
                else:
                    _total_tokens += sum(batch_tokens)
            self.total_tokens = _total_tokens

        if self.total_tokens > 0:
            if not self.count_prompt_tokens:
                # remove prompt tokens from count of generated tokens
                if hasattr(self, "prompt_tokens_lens"):
                    self.total_tokens -= sum(self.prompt_tokens_lens)
                else:
                    logger.info(
                        " Prompt tokens not removed from total tokens generated stats, 'prompt_tokens_lens' not set."
                    )

            self.perf_value = self.total_tokens / self.benchmark_duration
            self.perf_unit_str = "Tokens/sec"
        else:
            self.perf_value = self.total_samples / self.benchmark_duration
            self.perf_unit_str = "Samples/sec"
        return self.get_output_stats_dict()

    def get_output_stats_dict(self):
        output_stats_dict = {
            "total_run_time": self.benchmark_duration,
            "total_compilation_time": self.compilation_duration,
            "peak_host_mem_usage": self.peak_cpu_mem,
            "total_samples": self.total_samples,
            "samples_per_sec": self.total_samples / self.benchmark_duration,
            "tokens_per_sec": self.total_tokens / self.benchmark_duration,
            "inference_time_ms": self.benchmark_duration * 1000 / (self.total_samples),
            "evaluation_score": self.eval_score,
            "input_size": self.get_input_shape_str(),
            "device_name": self.device_name,
            "machine_name": self.machine_name,
            "benchmark_date": time.asctime(time.localtime(self.benchmark_end_time)),
            "host_device": self.host_device_name,
        }

        if self.device_type == DeviceType.TT:
            output_stats_dict["pybuda_hash"] = self.pybuda_hash
            output_stats_dict["pybuda_version_date"] = self.pybuda_hash_date
        else:
            output_stats_dict["pytorch_version"] = self.torch_version
            output_stats_dict["pybuda_version_date"] = "N/A"

        return output_stats_dict

    def print_output_stats(self):
        print("*****************************************************")
        print(" Device:", self.device_name)
        print(f" Total compilation time (s): {self.compilation_duration:.4f}")
        print(f" Peak host memory usage (MB): {self.peak_cpu_mem:.2f}")
        print(f" Total runtime (s) for {self.total_samples} inputs: {self.benchmark_duration:.4f}")
        print(f" {self.perf_unit_str}: {self.perf_value:.2f}")
        print(f" Inference time (ms): {(self.benchmark_duration / (self.total_samples)) * 1000:.1f}")
        if self.total_tokens > 0:
            print(f" Total tokens generated: {self.total_tokens}")
        print(f" Evaluation Score: {self.eval_score}")
        print(f" Batch size: {self.microbatch}")
        print(f" Input shape: {self.get_input_shape_str()}")
        print(f" Datatype: {self.dataformat}")
        print(" Host device:", self.host_device_name)
        if self.device_type == DeviceType.TT:
            print(" PyBUDA Version:", self.pybuda_hash)
            print(" PyBUDA Version Date:", self.pybuda_hash_date)
        else:
            print(" PyTorch Version:", self.torch_version)
        print("*****************************************************")
