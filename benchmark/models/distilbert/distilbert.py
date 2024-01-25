# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertModel, DistilBertTokenizer

from ...common import DummyNLPDataset, SST2Dataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["base"])
def distilbert(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

    # Set model parameters based on chosen task and model configuration
    if task == "na":
        if config == "base":
            model_name = "distilbert-base-uncased"
            seq_len = 128
            target_microbatch = 128
        else:
            raise RuntimeError("Unknown config")
    elif task == "text_classification":
        if config == "base":
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            seq_len = 128
            target_microbatch = 128
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Load model
        cfg = DistilBertConfig.from_pretrained(model_name)
        model = DistilBertModel(config=cfg)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_distilbert_na", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyNLPDataset(microbatch=microbatch, seq_len=seq_len, hidden_size=cfg.hidden_size)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "text_classification":

        # Load model and tokenizer
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_distilbert_text_classification", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Load SST-2 Dataset
        sst2_dataset = load_dataset("glue", "sst2")
        dataset = SST2Dataset(dataset=sst2_dataset, tokenizer=tokenizer, split="validation", seq_len=seq_len)

        # Define evaluation function
        def eval_fn(outputs, labels):
            import evaluate

            accuracy_metric = evaluate.load("accuracy")
            pred_labels = []
            true_labels = []
            for output in outputs:
                if device == "tt":
                    output = output[0].value()
                else:
                    output = output[0].detach().cpu()
                pred_labels.extend(torch.argmax(output, axis=-1))
            for label in labels:
                true_labels.extend(label)
            eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)

            return eval_score["accuracy"]

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
