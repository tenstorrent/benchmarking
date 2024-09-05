# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import json
import logging
import os
import pathlib
import pickle
import queue
import socket
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import torch
import transformers
from diffusers import StableDiffusionPipeline
from transformers.tokenization_utils_base import BatchEncoding

# Models
import benchmark.models.bert.bert
import benchmark.models.deit.deit
import benchmark.models.falcon.falcon
import benchmark.models.flant5.flant5
import benchmark.models.flant5.flant5_past_cache_enc_dec
import benchmark.models.hrnet.hrnet
import benchmark.models.inception_v4.inception_v4
import benchmark.models.mobilenet_v1.mobilenet_v1
import benchmark.models.mobilenet_v2.mobilenet_v2
import benchmark.models.mobilenet_v2.mobilenet_v2_timm
import benchmark.models.mobilenet_v3.mobilenet_v3
import benchmark.models.open_pose.open_pose
import benchmark.models.resnet.resnet
import benchmark.models.stable_diffusion.stable_diffusion
import benchmark.models.t5.t5
import benchmark.models.t5.t5_past_cache_enc_dec
import benchmark.models.unet.unet
import benchmark.models.vit.vit
import benchmark.models.vovnet.vovnet_v1
import benchmark.models.vovnet.vovnet_v2
import benchmark.models.whisper.whisper
import benchmark.models.whisper.whisper_enc_dec
import benchmark.models.yolo_v5.yolo_v5

# Common functions
from benchmark.common import get_models, store_model_output, torch_df_from_str
from benchmark.common.benchmark_run import BenchmarkRun

sys.path.append(".")

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def print_benchmark_envs():

    # If specified by env variable, print the environment variables
    # It can be useful in CI jobs to get the state of the enviroment variables before test session starts
    print_env_variables = bool(int(os.environ.get("PYTEST_PRINT_ENV_VARIABLES", "0")))
    if print_env_variables:
        pybuda_specific_vars = {}
        tt_backend_specific_vars = {}
        print(f"####### Environment variables - Count: {len(os.environ)} #######")
        for key, value in os.environ.items():
            print(f"{key}={value}")
            if key.startswith("PYBUDA_") or key.startswith("GOLDEN_"):
                pybuda_specific_vars[key] = value
            elif key.startswith("TT_BACKEND_"):
                tt_backend_specific_vars[key] = value

        print(f"####### PYBUDA specific enviroment variables - Count: {len(pybuda_specific_vars)} #######")
        for key, value in pybuda_specific_vars.items():
            print(f"{key}={value}")

        print(f"####### TT_BACKEND specific enviroment variables - Count: {len(tt_backend_specific_vars)} #######")
        for key, value in tt_backend_specific_vars.items():
            print(f"{key}={value}")


def run(
    args,
    model: Any,
    generator: torch.utils.data.DataLoader,
    eval_fn: Callable,
    benchmark_run: BenchmarkRun,
) -> Dict[str, Any]:

    # Define input and output lists
    input_data = []
    store_outputs = []
    store_labels = []
    for batch, labels in generator:
        if isinstance(model, transformers.pipelines.Pipeline):
            input_data.append((list(batch), labels))
        else:
            input_data.append((batch, labels))
    benchmark_run.set_input_shape(input_data, model)

    if args.device == "tt":
        # Import PyBUDA packages
        import pybuda
        from pybuda._C.backend_api import BackendDevice
        from pybuda.tools.tti_data_parallel import (
            ForwardInputs,
            GenerativeInputs,
            RunMode,
            RunResult,
            initialize_multicard_runner,
            split_tensor_batch,
        )

        from benchmark.common import df_from_str, mf_from_str, trace_from_str

        def data_parallel_tti_run(
            arch: BackendDevice,
            inputs: List[Any],
            loop_count: int,
            output_dir: str,
            device_ids: List[int],
            benchmark_run: BenchmarkRun,
            tt_device: Optional[pybuda.TTDevice] = None,
            precompiled_image_path: Optional[str] = None,
        ):
            def duplicate_batch(input_data, factor: int):

                def to_list(data):
                    if isinstance(data, tuple) or isinstance(data, list):
                        return [to_list(item) for item in data]
                    else:
                        return data

                def duplicate_tensor(data, factor: int):
                    if isinstance(data, torch.Tensor):
                        duplicated_tensor = torch.cat([data] * factor, dim=0)
                        return duplicated_tensor

                    elif isinstance(data, list):
                        for i in range(len(data)):
                            data[i] = duplicate_tensor(data[i], factor)

                    else:
                        raise TypeError("Input data should contain list or torch tensor only")

                    return data

                data = to_list(input_data)
                return duplicate_tensor(data, factor)

            assert tt_device or precompiled_image_path, "One of tt_device or precompiled_image_path must be specified"
            assert not (tt_device and precompiled_image_path), "Specify one of tt_device, precompiled_image_path"
            num_devices = len(device_ids)

            # Get compilation sample inputs
            sample_inputs = inputs[0][0]
            if isinstance(sample_inputs, dict) or isinstance(
                sample_inputs, transformers.tokenization_utils_base.BatchEncoding
            ):
                sample_inputs = list(sample_inputs.values())
            if isinstance(model, dict):
                if benchmark_run.has_compile_inputs:
                    sample_inputs = model["compile_inputs"]

            if benchmark_run.has_forward_wrapper:
                run_mode = RunMode.GENERATIVE
                # Duplicate the predefined microbatch size (always 1) by the number of devices for generative run
                compile_inputs = duplicate_batch(sample_inputs, len(device_list))
            else:
                run_mode = RunMode.FORWARD
                compile_inputs = sample_inputs

            if tt_device is not None:
                image_path = os.path.join(output_dir, "parallel_tti_run.tti")
                single_device_inputs = split_tensor_batch(compile_inputs, num_devices)[0]
                monitor_thread = threading.Thread(target=benchmark_run.cpu_usage_monitor)
                monitor_thread.start()
                benchmark_run.start_compilation_timer()
                tt_device.compile_to_image(img_path=image_path, training=False, sample_inputs=single_device_inputs)
                benchmark_run.stop_monitoring = True
                benchmark_run.end_compilation_timer()
                monitor_thread.join()
            elif precompiled_image_path is not None:
                image_path = precompiled_image_path

            pybuda.pybuda_reset()

            # Provide an input sample for compilation (i.e the input for 1 loop)
            runner = initialize_multicard_runner(
                arch=arch,
                device_ids=device_ids,
                run_mode=run_mode,
                compile_inputs=compile_inputs,
                precompiled_tti_path=image_path,
                output_dir=output_dir,
            )

            benchmark_run.start_benchmark_timer()
            for loop in range(loop_count):
                for batch, labels in inputs:
                    if isinstance(batch, dict) or isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                        batch = list(batch.values())
                    if run_mode == RunMode.FORWARD:
                        run_result: RunResult = runner.run(ForwardInputs(run_inputs=batch))
                    else:
                        batch = duplicate_batch(batch, len(device_list))
                        run_result: RunResult = runner.run(
                            GenerativeInputs(
                                run_inputs=batch,
                                num_tokens_to_generate=64,
                                write_index=0,
                                first_current_index=32,
                                pad_token_id=None,
                            )
                        )
                    output = [
                        pybuda.tensor.Tensor.create_from_torch(output_tensor) for output_tensor in run_result.outputs[0]
                    ]

                    if loop == 0:
                        store_labels.append(labels)
                        store_outputs.append(output)
            benchmark_run.end_benchmark_timer()
            runner.shutdown()

            return store_outputs

        # Set default configuration type
        pybuda.config.set_configuration_options(default_df_override=df_from_str(args.dataformat))
        if args.acc_dataformat:
            pybuda.config.set_configuration_options(accumulate_df=df_from_str(args.acc_dataformat))

        if args.dump_intermediate:
            ops = args.dump_intermediate.split(",")
            pybuda.set_configuration_options(op_intermediates_to_save=ops)

        # Override push timeout on slow runs
        os.environ["TT_BACKEND_PUSH_TIMEOUT"] = "600"
        os.environ["TT_BACKEND_TIMEOUT"] = "1200"

        # TODO: For silicon device runs, it seems that the `tt` from user-side is not
        # the one being used with api calls like pybuda.run_forward(..). We'll fetch
        # the arch from the first device-type available
        device_list = pybuda.detect_available_devices()
        if len(device_list) == 0:
            raise RuntimeError("No Tenstorrent devices found. pybuda.detect_available_devices() returns no devices.")

        if args.chips == 0:
            raise RuntimeError(f"args.chips:={args.chips}, this is an error and not supported in Benchmarking.")
        elif args.chips > 1:
            logger.warning(
                f" WARNING args.chips:={args.chips} is attempting to run multichip. This may not be supported."
            )

        logger.info(f"args.chips:={args.chips}")
        arch = device_list[0]

        # Place device modules
        if isinstance(model, dict):
            assert "device" in model
            if args.load_tti:
                print(f"Loading TTDevice from TTI specified at: {args.load_tti}")
                img = pybuda.TTDeviceImage.load_from_disk(args.load_tti)
                img.info()
                device = pybuda.TTDevice.load_image(img=img)
            else:
                if args.save_tti or (args.parallel_tti and not args.load_tti):
                    print(f"Saving TTDevice Image to: {args.save_tti}")
                device = pybuda.TTDevice(
                    "tt0",
                    module=model["device"],
                    fp32_fallback=df_from_str(args.dataformat),
                    num_chips=args.chips,
                    arch=arch,
                )
        elif args.save_tti or args.load_tti:
            raise Exception(f"{args.model} currently cannot be compiled to TTI.")

        # Enable perf analysis, if selected
        if args.perf_analysis:
            args.loop_count = 1
            if "PYBUDA_OP_PERF" not in os.environ:
                os.environ["PYBUDA_OP_PERF"] = "1"
            if "TT_BACKEND_PERF_ANALYZER" not in os.environ:
                os.environ["TT_BACKEND_PERF_ANALYZER"] = "1"

        # Set PyBUDA configurations
        pybuda.set_configuration_options(
            math_fidelity=mf_from_str(args.math_fidelity),
            performance_trace=trace_from_str(args.trace),
            backend_opt_level=args.backend_opt_level,
            enable_recompute=args.recompute,
            enable_auto_transposing_placement=args.auto_transpose,
        )

        # Get compilation sample inputs
        sample_inputs, targets = input_data[0][0], input_data[0][1]
        if isinstance(sample_inputs, dict) or isinstance(
            sample_inputs, transformers.tokenization_utils_base.BatchEncoding
        ):
            sample_inputs = list(sample_inputs.values())
        if isinstance(model, dict):
            if benchmark_run.has_compile_inputs:
                sample_inputs = model["compile_inputs"]

        # Set targets for training, if selected
        if args.training:
            targets = [targets]
            assert len(targets) > 0, "Targets must be supplied for training"
        else:
            targets = tuple()

        # Compile model
        if not args.parallel_tti:
            if isinstance(model, dict):
                if args.save_tti:
                    device.compile_to_image(
                        img_path=args.save_tti,
                        training=args.training,
                        sample_inputs=sample_inputs,
                        sample_targets=targets,
                    )
                    print(f"Pybuda successfully compiled model to: {args.save_tti}")
                    exit(0)

                if "verify_cfg" in model.keys():
                    verify_cfg = model["verify_cfg"]
                else:
                    verify_cfg = pybuda.verify.VerifyConfig(verify_pybuda_codegen_vs_framework=True)

                # Compilation run
                monitor_thread = threading.Thread(target=benchmark_run.cpu_usage_monitor)
                monitor_thread.start()
                benchmark_run.start_compilation_timer()
                output_q = pybuda.initialize_pipeline(
                    training=args.training,
                    sample_inputs=sample_inputs,
                    _verify_cfg=verify_cfg,
                    sample_targets=targets,
                )
                benchmark_run.stop_monitoring = True
                benchmark_run.end_compilation_timer()
                monitor_thread.join()
                if not benchmark_run.has_forward_wrapper:
                    # Prepare a thread pushing inputs
                    def push_inputs_thread():
                        for loop in range(args.loop_count):
                            for batch, labels in input_data:
                                if pybuda.error_raised():
                                    print(" * Aborting input thread due to error")
                                    return
                                if isinstance(batch, dict):
                                    device.push_to_inputs(list(batch.values()))
                                else:
                                    device.push_to_inputs(batch)
                                if loop == 0:
                                    store_labels.append(labels)

                    input_thread = threading.Thread(target=push_inputs_thread)

                    # Prepare a threading popping outputs
                    def pop_outputs_thread(output_q):
                        if args.dump_intermediate:
                            intermediates_queue = pybuda.get_intermediates_queue()
                            torch.set_printoptions(
                                threshold=100000000,
                                linewidth=300,
                                precision=4,
                                sci_mode=False,
                            )
                        number_of_samples = len(generator)
                        for i in range(args.loop_count * number_of_samples):
                            while True:
                                try:
                                    output = output_q.get()
                                    if args.dump_intermediate:
                                        intermed = intermediates_queue.get()
                                        if i < args.dump_intermediate_count:
                                            # Dump text log file for human consumption
                                            with open(
                                                f"intermed_{args.dump_intermediate_tag}.log",
                                                "w",
                                            ) as f:
                                                for tns in intermed:
                                                    f.write(f"intermed input {i}: len={len(intermed)} \n")
                                                    f.write(f"{tns}\n")
                                            # Pickle the array of intermediate values and write to a binary file to be checked off-line
                                            with open(
                                                f"intermed_{args.dump_intermediate_tag}.p",
                                                "wb",
                                            ) as f:
                                                pickle.dump(intermed, f)

                                    if i < number_of_samples:
                                        store_outputs.append(output)
                                    break  # got data, break out of forever loop
                                except queue.Empty:
                                    if pybuda.error_raised():
                                        print(" * Aborting output thread due to error")
                                        return

                    # Set output
                    output = output_q if not args.training else pybuda.get_loss_queue()
                    output_thread = threading.Thread(target=pop_outputs_thread, args=(output,))
                    output_thread.start()

                    # Sync - Make sure all process setup, compile, etc. is done
                    pybuda.sync()

                    # Start input thread
                    input_thread.start()
                    time.sleep(2)  # Let the input thread start up and transfer initial data
            else:
                # Compilation loop for pybuda_pipeline models
                monitor_thread = threading.Thread(target=benchmark_run.cpu_usage_monitor)
                monitor_thread.start()
                benchmark_run.start_compilation_timer()
                sample_inputs = input_data[0][0]
                _ = model(sample_inputs, batch_size=args.microbatch)
                benchmark_run.stop_monitoring = True
                benchmark_run.end_compilation_timer()
                monitor_thread.join()

    # Run benchmark
    if args.device == "tt" and isinstance(model, dict):
        if args.parallel_tti:
            assert not args.training, "Training not supported in parallel tti run"
            assert args.chips == 1, "Parallel TTI only supported for single chip models"
            assert len(device_list) > 0
            device_ids = [[i] for i in range(len(device_list))]
            precompiled_image_path = args.load_tti if args.load_tti else None
            tt_device = None if args.load_tti else device

            store_outputs = data_parallel_tti_run(
                arch=arch,
                inputs=input_data,
                loop_count=args.loop_count,
                output_dir=args.parallel_tti,
                device_ids=device_ids,
                benchmark_run=benchmark_run,
                tt_device=tt_device,
                precompiled_image_path=precompiled_image_path,
            )

        else:
            benchmark_run.start_benchmark_timer()
            if benchmark_run.has_forward_wrapper:
                for loop in range(args.loop_count):
                    for batch, labels in input_data:
                        output = model["forward_wrapper"](
                            batch=batch,
                            output_q=output_q,
                            device=device,
                        )
                        if loop == 0:
                            store_labels.append(labels)
                            store_outputs.append(output)
            else:
                pybuda.run_forward(input_count=(args.loop_count * len(generator)))
                input_thread.join()
                output_thread.join()

            benchmark_run.end_benchmark_timer()

        # Combine outputs for data parallel runs
        if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
            flattened_list = [[tensor] for sublist in store_outputs for tensor in sublist]
            store_outputs = flattened_list

        if pybuda.error_raised():
            print("*********************************")
            print(" Error raised, aborting benchmark")
            print("*********************************")
            return {
                "total_run_time": 0,
                "total_samples": 0,
                "samples_per_sec": 0,
                "evaluation_score": 0,
                "args": vars(args),
                "arch": str(arch).split(".", 1)[1],
                "machine_name": benchmark_run.machine_name,
                "pybuda_hash": benchmark_run.pybuda_hash,
            }
    else:
        # cuda, CPU, or TT device with pybuda_pipeline implementations
        benchmark_run.start_benchmark_timer()
        with torch.inference_mode():
            for loop in range(args.loop_count):
                for batch, labels in input_data:
                    if isinstance(batch, dict) or isinstance(batch, transformers.tokenization_utils_base.BatchEncoding):
                        batch = BatchEncoding(batch).to(args.device)
                        output = model(**batch)
                    else:
                        if isinstance(model, transformers.pipelines.Pipeline):
                            output = model(batch, batch_size=args.microbatch)
                        elif isinstance(model, StableDiffusionPipeline):
                            output = model(prompt=batch, num_images_per_prompt=1, output_type="pil").images
                        elif hasattr(batch[0], "to"):
                            output = model(
                                batch[0].to(
                                    args.device,
                                    dtype=torch_df_from_str(args.dataformat),
                                )
                            )
                        else:
                            output = model(batch)
                    if loop == 0:
                        store_labels.append(labels)
                        store_outputs.append(output)
        benchmark_run.end_benchmark_timer()

    # Store model output
    if args.model_output:
        store_model_output(model, benchmark_run, store_outputs, store_labels)

    # Benchmark results
    eval_score = eval_fn(outputs=store_outputs, labels=store_labels)
    output_stats_dict = benchmark_run.calc_output_stats(store_outputs, model, eval_score)
    benchmark_run.print_output_stats()
    return output_stats_dict


if __name__ == "__main__":

    print_benchmark_envs()

    # Arguments
    parser = argparse.ArgumentParser(description="Benchmark a model on TT hardware")
    parser.add_argument("-m", "--model", help="Model to benchmark (i.e. bert)")
    parser.add_argument(
        "-c",
        "--config",
        help="Model configuration to benchmark (i.e. tiny, base, large)",
    )
    parser.add_argument("--task", help="Model task (i.e. na, text_classification)")
    parser.add_argument("-d", "--device", help="Device to benchmark (i.e. tt, cpu, cuda)", default="tt")
    parser.add_argument("-t", "--training", action="store_true", help="Benchmark training")
    parser.add_argument(
        "-df",
        "--dataformat",
        choices=["Fp32", "Fp16", "Fp16_b", "Bfp8", "Bfp8_b", "Bfp4", "Bfp4_b"],
        default="Fp16_b",
        help="Set data format",
    )
    parser.add_argument(
        "-adf",
        "--acc_dataformat",
        choices=["Fp32", "Fp16", "Fp16_b", "Bfp8", "Bfp8_b", "Bfp4", "Bfp4_b"],
        help="Set accumulation data format",
    )
    parser.add_argument(
        "-mf",
        "--math_fidelity",
        choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"],
        default="HiFi3",
        help="Set math fidelity",
    )
    parser.add_argument(
        "-opt",
        "--backend_opt_level",
        choices=[0, 1, 2, 3, 4],
        default=3,
        type=int,
        help="Set backend optimization level",
    )
    parser.add_argument(
        "--loop_count",
        default=1,
        type=int,
        help="Set the number of times to loop through the model.",
    )
    parser.add_argument(
        "-mb",
        "--microbatch",
        default=1,
        type=int,
        help="The microbatch size to run the benchmark on.",
    )
    parser.add_argument(
        "--num_tokens",
        default=1,
        type=int,
        help="The number of tokens to run text generation models only.",
    )
    parser.add_argument("--chips", default=1, type=int, help="Number of chips to run benchmark on.")
    parser.add_argument("--recompute", action="store_true", help="Enable recompute in training")
    parser.add_argument(
        "--trace",
        default="none",
        choices=["none", "light", "verbose"],
        help="Performance trace to be generated during the run.",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available models and configurations",
    )
    parser.add_argument(
        "-e",
        "--env",
        default="",
        help='List of environment variable settings, i.e. "PYBUDA_OPT1=1 PYBUDA_OP2=1" to run with.',
    )
    parser.add_argument(
        "--save_output",
        action="store_true",
        help="Output json file",
    )
    parser.add_argument(
        "--auto_transpose",
        action="store_true",
        help="Enable auto-transpose on placement",
    )
    parser.add_argument(
        "--dump_intermediate",
        "-di",
        help="List intermediate ops whose values should be dumped into log/pickle files",
    )
    parser.add_argument("--dump_intermediate_tag", "-dit", help="Filename tag to be used for this run")
    parser.add_argument(
        "--dump_intermediate_count",
        "-dic",
        type=int,
        default=1,
        help="The number of inputs to dump",
    )
    parser.add_argument(
        "--load_tti",
        type=str,
        help="Skip compile and load from TTI-archive configured for silicon (specify path to TTI).",
    )
    parser.add_argument(
        "--save_tti",
        type=str,
        help="Save compilation for TTDevice into a TTI-archive configured for silicon to file and exit program. (specify path to save to).",
    )
    parser.add_argument(
        "--perf_analysis", action="store_true", help="Enable backend perf analyzer and op estimates in compiler"
    )
    parser.add_argument(
        "--model_output",
        action="store_true",
        help="Store samples and model output per sample in text file for debugging.",
    )
    parser.add_argument(
        "--parallel_tti",
        default="",
        type=str,
        help="Save compilation for TTDevice into a TTI-archive configured for silicon to file and run it in parallel. (specify directory path to save archive and dump outputs).",
    )
    args = parser.parse_args()

    # Get all available models
    models = get_models()

    # Process arguments
    if args.list:
        print("\nAvailable models:\n")
        for m in models:
            print(" - ", m.ljust(30), "configs: ", models[m]["configs"])
        print("\n")
        exit(0)

    if not args.model:
        print("\nModel must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    if args.model not in models:
        print("Invalid model name. Available models: ")
        print(list(models.keys()))
        exit(1)

    if args.config:
        if args.config not in models[args.model]["configs"]:
            print(
                "Invalid configuration for model ",
                args.model,
                ". Available configurations:",
            )
            print(models[args.model]["configs"])
            exit(1)
    elif len(models[args.model]["configs"]) > 1:
        print(
            "Model ",
            args.model,
            " has more than one configuration, you have to choose one:",
        )
        print(models[args.model]["configs"])
        exit(1)

    if sum([bool(args.save_tti), bool(args.parallel_tti)]) > 1:
        print("Specify only one of `--save_tti`, `--parallel_tti`")
        exit(1)

    if args.parallel_tti:
        assert not args.training, "Training not supported in parallel tti run"
        print("Overriding args.chips to 1 since parallel_tti is set")
        args.chips = 1
        print(f"Saving TTDevice Image and output artefacts to: {args.parallel_tti}")

    if args.env != "":
        envs = args.env.split(" ")
        for e in envs:
            if "=" not in e:
                name = e
                value = "1"
            else:
                name, value = e.split("=")
            os.environ[name] = value

    # Load model and run benchmark
    kwargs = {
        "training": args.training,
        "microbatch": args.microbatch,
        "task": args.task,
        "device": args.device,
        "data_type": args.dataformat,
    }
    func = models[args.model]["func"]
    available_parameters = inspect.signature(func).parameters
    for p in available_parameters:
        if p == "config":
            if args.config is None:
                assert len(models[args.model]["configs"]) == 1
                kwargs["config"] = models[args.model]["configs"][0]
            else:
                kwargs["config"] = args.config
    benchmark_run = BenchmarkRun(args=args)
    logger.info(f" creating benchmarking run: {func.__name__}")
    logger.info(f" kwargs: {kwargs}")
    model, generator, eval_fn = models[args.model]["func"](benchmark_run=benchmark_run, **kwargs)
    error = False
    try:
        result = run(args, model, generator, eval_fn, benchmark_run)
    except RuntimeError as e:
        result = {
            "args": vars(args),
            "samples_per_sec": 0.0,
            "error": str(e),
            "machine_name": socket.gethostname(),
        }
        print("Error encountered while running benchmark: ", e)
        traceback.print_exc()
        error = True

    # Store outputs
    if args.save_output:
        result.update(vars(args))
        fname = f"perf_{args.model}_{args.config}_{result.get('input_size', 'na')}_{args.device}_mb{args.microbatch}_{benchmark_run.short_run_id}.json"
        fname = fname.replace("/", "_")  # escape fnames
        out_file = pathlib.Path("results", fname)

        # Creates result dir if models are run out of the benchmarking repo
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        all_results = []
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    print("Reading in ", out_file, " with previous data")
                    all_results = json.loads(f.read())
            except Exception as e:
                print(
                    f"{str(e)}: Failed to load previous results, Will not overwrite, but create a different output file."
                )
                out_file = "post_error_" + out_file

        all_results.append(result)
        with open(out_file, "w") as f:
            f.write(json.dumps(all_results))

        print("Written out ", out_file, " with summary")

    if error:
        exit(2)
