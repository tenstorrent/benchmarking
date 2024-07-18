# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from ...common import DummyPipelineDataset, LibriSpeechDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["tiny", "base", "small", "medium", "large"])
def whisper(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):
    if device == "tt":
        import pybuda
        from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_t_streaming = True

    # Set model parameters based on chosen task and model configuration
    if task == "na" or task == "asr":
        if config == "tiny":
            model_name = "openai/whisper-tiny"
        elif config == "base":
            model_name = "openai/whisper-base"
        elif config == "small":
            model_name = "openai/whisper-small"
        elif config == "medium":
            model_name = "openai/whisper-medium"
        elif config == "large":
            model_name = "openai/whisper-large-v2"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    if task == "na":
        max_new_tokens = 69
    elif task == "asr":
        max_new_tokens = 256

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Configure model mode for training or evaluation
    if training:
        model.train()
    else:
        model.eval()

    # Create model device placement map
    if device == "tt":
        model = pybuda_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
        )
    else:
        device = 0 if device == "cuda" else -1
        model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            torch_dtype=torch_df_from_str(data_type),
        )

    # set model parameters
    model.model.config.max_length = max_new_tokens
    model.model.config.early_stopping = True
    # set key for accessing output text
    model.output_key = "text"
    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # Task specific configuration
    if task == "na":

        # set task specific model parameters
        model.model.config.early_stopping = False

        # Get sample
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        sample_audio = ds[0]["audio"]["array"]

        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text=sample_audio,
            answer="",
        )
        collate_fn = None

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "asr":

        # for full validation `librispeech_asr` split="test.clean"
        # contains 2620 audio samples
        # ds = load_dataset("librispeech_asr", "clean", split="test")
        # for quicker testing `hf-internal-testing/librispeech_asr_dummy`
        # split="validation.clean" contains 73 audio samples
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        # create dataset with handling class
        dataset = LibriSpeechDataset(
            dataset=ds,
        )

        # librispeech_asr has samples of different lengths, collate_fn is required
        def collate_fn(batch):
            # This collate_fn separates the data and labels
            data = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            return data, labels

        # Define evaluation function
        def eval_fn(outputs=None, labels=None, **kwargs):
            # labels are all upper case, make outputs upper case to match
            post_outputs = [out["text"].strip().upper() for batch_outputs in outputs for out in batch_outputs]
            # labels is lists of lists, flatten them
            # there also may not be "." at the end of each label
            flattened_labels = [lbl if lbl[-1] == "." else lbl + "." for batch_labels in labels for lbl in batch_labels]
            # calculate Word Error Rate (WER)
            # see: https://huggingface.co/spaces/evaluate-metric/wer
            wer = load("wer")
            wer_score = wer.compute(predictions=post_outputs, references=flattened_labels)
            # word accuracy score (positive is better) is complement of error score
            # W_acc = 1 - WER = (Correct - Inserted) / (Number of words in the reference)
            eval_score = 1 - wer_score
            return eval_score

    # Create DataLoader
    generator = DataLoader(
        dataset,
        batch_size=microbatch,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
