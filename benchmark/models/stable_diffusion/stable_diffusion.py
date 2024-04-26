# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import re

from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor

from ...common import BenchmarkRun, DummyPipelineDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["v1-4"])
def stable_diffusion(
    training: bool, task: str, config: str, microbatch: int, device: str, data_type: str, benchmark_run: BenchmarkRun
):
    import pybuda

    compiler_cfg = pybuda.config._get_global_compiler_config()

    if compiler_cfg.balancer_policy == "default":
        compiler_cfg.balancer_policy = "Ribbon"
        os.environ["PYBUDA_RIBBON2"] = "1"

    os.environ["ROUTER_SKIP_RESOURCE_VALIDATION_CHECK"] = "1"

    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{14*1024}"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
    os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
    os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
    os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"

    pybuda.config.override_op_size("layernorm_2036.dc.multiply.4", (2, 1))
    pybuda.config.override_op_size("layernorm_2196.dc.multiply.4", (2, 1))
    pybuda.config.override_op_size("matmul_1878", (2, 1))
    pybuda.config.override_op_size("matmul_1884", (2, 1))
    pybuda.config.override_op_size("matmul_1893", (2, 1))

    # Set model parameters based on chosen task and model configuration
    if task in ["na", "image_generation"]:
        if config == "v1-4":
            model_name = "CompVis/stable-diffusion-v1-4"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    if task == "na":
        num_inference_steps = 50
    elif task == "image_generation":
        num_inference_steps = 50

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    if device == "tt":
        from benchmark.models.stable_diffusion.stable_diffusion_impl import (
            create_compile_input,
            denoising_loop,
            generate_model_stable_diffusion,
            stable_diffusion_postprocessing,
            stable_diffusion_preprocessing,
        )

        # load model
        tt_module, pipe = generate_model_stable_diffusion(model_name)

        def forward_wrapper(
            batch,
            output_q,
            device,
            pipe=pipe,
            num_inference_steps=num_inference_steps,
        ):
            inputs = batch[0]
            (
                latents,
                timesteps,
                prompt_embeds,
                extra_step_kwargs,
            ) = inputs
            # Run inference on TT device
            latents = denoising_loop(
                output_q,
                pipe,
                latents,
                timesteps,
                prompt_embeds,
                extra_step_kwargs,
                num_inference_steps=num_inference_steps,
                ttdevice=device,
            )
            post_proc = stable_diffusion_postprocessing(pipeline=pipe, latents=latents)
            batch_output = post_proc.images
            return batch_output

        model = {"device": tt_module, "forward_wrapper": forward_wrapper}
    elif device == "cuda":
        model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_df_from_str(data_type)).to("cuda")
    elif device == "cpu":
        model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_df_from_str(data_type))
    else:
        raise RuntimeError("Unknown device")

    def preprocessed_collate_fn(batch):
        # Separate inputs and labels
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return inputs, labels

    # Task specific configuration
    if task == "na":
        # Create test dataset
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="An image of a cat",
            answer="An image of a cat",
        )

        def eval_fn(outputs, labels):
            return 0

    elif task == "image_generation":

        # create test dataset
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="",
            answer="",
        )
        # list of prompts
        prompts = [
            "a photo of an astronaut riding a horse on mars",
            "A high tech solarpunk utopia in the Amazon rainforest",
            "A pikachu fine dining with a view to the Eiffel Tower",
            "A mecha robot in a favela in expressionist style",
            "an insect robot preparing a delicious meal",
            "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        ]

        # set prompt data
        dataset.data = [
            (
                prompt,
                prompt,
            )
            for prompt in prompts
        ]

        # Define evaluation function
        def eval_fn(outputs, labels):
            # Data post-processing
            img_tensors = [pil_to_tensor(img) for b_out in outputs for img in b_out]
            flat_labels = [lab for b_labs in labels for lab in b_labs]
            # https://torchmetrics.readthedocs.io/en/stable/multimodal/clip_score.html#torchmetrics.multimodal.clip_score.CLIPScore
            # If a single tensor it should have shape ``(N, C, H, W)``. If a list of tensors, each tensor should have shape
            # ``(C, H, W)``. ``C`` is the number of channels, ``H`` and ``W`` are the height and width of the image.
            metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
            eval_score = metric(img_tensors, flat_labels).detach().mean().item()
            return eval_score

    # define store_output_func so we can store outputs in a human readable format
    def store_output_func(model, benchmark_run, output, labels, output_dir):
        output_image_dir = output_dir.joinpath(benchmark_run.run_name)
        output_image_dir.mkdir(parents=True, exist_ok=True)
        for b_idx, (b_out, b_lab) in enumerate(zip(output, labels)):
            for idx, (img, lab) in enumerate(zip(b_out, b_lab)):
                # save images with leading batch idx and img idx as well as prompt
                img_name = re.sub(r"[^a-zA-Z0-9_]", "", lab.replace(" ", "_"))
                fname = f"img_{b_idx}_{idx}_{img_name}.png"
                img_out_fpath = output_image_dir.joinpath(fname)
                img.save(img_out_fpath)

    benchmark_run.store_output_func = store_output_func
    if device == "tt":
        dataset.data = [
            (
                stable_diffusion_preprocessing(
                    pipe,
                    prompt,
                    num_inference_steps=num_inference_steps,
                ),
                label,
            )
            for (prompt, label) in dataset.data
        ]
        # create compile_input
        # NOTE: in the general case compile input may look different from the per sample
        # required input. This can be done more efficiently and much processing can be avoided.
        model["compile_inputs"] = create_compile_input(pipeline=pipe)

    # Create DataLoader
    generator = DataLoader(
        dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=preprocessed_collate_fn
    )

    return model, generator, eval_fn
