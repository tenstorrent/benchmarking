# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import importlib.util
import inspect
import logging
from pathlib import Path

import torch
import transformers

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = {}


# decorator
def benchmark_model(configs=[]):
    def benchmark_decorator(model_func):
        name = model_func.__name__
        global MODELS

        @functools.wraps(model_func)
        def wrapper(*args, benchmark_run=None, **kwargs):
            func_params = inspect.signature(model_func).parameters
            if "benchmark_run" in func_params:
                # pass benchmarkrun obj to model_func for specialization
                model, generator, eval_fn = model_func(*args, benchmark_run=benchmark_run, **kwargs)
            else:
                # default configuration of benchmarkrun obj
                model, generator, eval_fn = model_func(*args, **kwargs)
            # benchmark_run gets metadata from model implementation
            benchmark_run.default_model_init(model_func, model, generator, eval_fn)
            return model, generator, eval_fn

        MODELS[name] = {"func": wrapper, "configs": configs}
        return wrapper

    return benchmark_decorator


def torch_df_from_str(df: str) -> torch.dtype:

    if df == "Fp64":
        return torch.float64

    if df == "Fp32":
        return torch.float32

    if df == "Fp16":
        return torch.float16

    if df == "Fp16_b":
        return torch.bfloat16

    raise RuntimeError("Unknown format: " + df)


if importlib.util.find_spec("pybuda"):
    import pybuda

    def df_from_str(df: str) -> pybuda.DataFormat:
        if df == "Fp32":
            return pybuda.DataFormat.Float32

        if df == "Fp16":
            return pybuda.DataFormat.Float16

        if df == "Fp16_b":
            return pybuda.DataFormat.Float16_b

        if df == "Bfp8":
            return pybuda.DataFormat.Bfp8

        if df == "Bfp8_b":
            return pybuda.DataFormat.Bfp8_b

        if df == "Bfp4":
            return pybuda.DataFormat.Bfp4

        if df == "Bfp4_b":
            return pybuda.DataFormat.Bfp4_b

        raise RuntimeError("Unknown format: " + df)

    def mf_from_str(mf: str) -> pybuda.MathFidelity:

        if mf == "LoFi":
            return pybuda.MathFidelity.LoFi

        if mf == "HiFi2":
            return pybuda.MathFidelity.HiFi2

        if mf == "HiFi3":
            return pybuda.MathFidelity.HiFi3

        if mf == "HiFi4":
            return pybuda.MathFidelity.HiFi4

        raise RuntimeError("Unknown math fidelity: " + mf)

    def trace_from_str(trace: str) -> pybuda.PerfTraceLevel:

        if trace == "none":
            return pybuda.PerfTraceLevel.NONE

        if trace == "light":
            return pybuda.PerfTraceLevel.LIGHT

        if trace == "verbose":
            return pybuda.PerfTraceLevel.VERBOSE

        raise RuntimeError("Unknown trace type: " + trace)


def get_num_tokens_generated(batch_output, tokenizer, key, output_has_eos=True):
    # assumes that samples given generated a eos_token that is not in text
    num_tokens = [
        (len(tokenizer.tokenize(out[key])) if isinstance(out, dict) else len(tokenizer.tokenize(out[0][key])))
        + int(output_has_eos)
        for out in batch_output
    ]
    return num_tokens


def get_model_output_dir():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir_name = "model_output"
    output_dir = Path(project_root, output_dir_name)
    return output_dir


def store_text_tokens(model, benchmark_run, output, labels, output_dir: Path):
    fname = f"{benchmark_run.run_name}_model_output.txt"
    output_fpath = output_dir.joinpath(fname)
    # get num_tokens
    if isinstance(model, transformers.pipelines.Pipeline):
        num_tokens = [get_num_tokens_generated(b_out, model.tokenizer, model.output_key) for b_out in output]
    else:
        num_tokens = [[len(out) for out in b_out] for b_out in output]
    lines_str = []
    for b_idx, (b_out, b_lab) in enumerate(zip(output, labels)):
        lines_str.append(f"\nbatch: {b_idx}\n")
        for s_idx, (out, lab) in enumerate(zip(b_out, b_lab)):
            lines_str.append(f"sample: {s_idx}\n")
            if isinstance(model, transformers.pipelines.Pipeline):
                lines_str.append(f"output:\n{out[model.output_key]}\n")
            else:
                lines_str.append(f"output:\n{model['tokenizer'].decode(out)}\n")
            lines_str.append(f"num_tokens: {num_tokens[b_idx][s_idx]}\n")
            lines_str.append(f"label:\n{lab}\n")
    logger.info(f" Writing model output to: {output_fpath}")
    with open(output_fpath, "w") as file:
        file.writelines(lines_str)


def store_model_output(model, benchmark_run, output, labels):
    if not (benchmark_run.has_forward_wrapper or isinstance(model, transformers.pipelines.Pipeline)):
        logging.warning(" Storing model output not implemented, skipping.")
        return
    output_dir = get_model_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    if benchmark_run.store_output_func is not None:
        # stable diffusion has a custom store_output_func
        benchmark_run.store_output_func(model, benchmark_run, output, labels, output_dir)
    else:
        store_text_tokens(model, benchmark_run, output, labels, output_dir)


def get_models():
    return MODELS
