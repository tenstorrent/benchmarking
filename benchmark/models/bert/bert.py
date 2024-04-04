# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification, BertModel, BertTokenizer

from ...common import DummyNLPDataset, SST2Dataset, StackExchangeDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["tiny", "base", "large"])
def bert(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda
        from pybuda._C.backend_api import BackendDevice

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.enable_auto_transposing_placement = True
        if compiler_cfg.balancer_policy == "default":
            compiler_cfg.balancer_policy = "Ribbon"
            os.environ["PYBUDA_RIBBON2"] = "1"
            os.environ["PYBUDA_TEMP_ENABLE_NEW_FUSED_ESTIMATES"] = "1"
            os.environ["PYBUDA_TEMP_ENABLE_NEW_SPARSE_ESTIMATES"] = "1"
            os.environ["PYBUDA_TEMP_SCALE_SPARSE_ESTIMATE_ARGS"] = "1"
            os.environ["PYBUDA_RIBBON2_CALCULATE_TARGET_CYCLES"] = "1"
            os.environ["PYBUDA_EXP_APPROX"] = "1"
            if data_type == "Bfp8_b":
                pybuda.config.configure_mixed_precision(op_type="add", output_df=pybuda.DataFormat.Float16_b)
                pybuda.config.configure_mixed_precision(op_type="subtract", output_df=pybuda.DataFormat.Float16_b)
                pybuda.config.configure_mixed_precision(op_type="reciprocal", output_df=pybuda.DataFormat.Float16_b)

        available_devices = pybuda.detect_available_devices()
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{18*1024}"
            if config == "large":
                pybuda.config.override_op_size("gelu_103", (3, 1))

    # Set model parameters based on chosen task and model configuration
    if task == "na":
        if config == "tiny":
            model_name = "prajjwal1/bert-tiny"
            seq_len = 128
            target_microbatch = 512
        elif config == "base":
            model_name = "bert-base-uncased"
            seq_len = 128
            target_microbatch = 128
        elif config == "large":
            model_name = "bert-large-uncased"
            seq_len = 384
            target_microbatch = 128
        else:
            raise RuntimeError("Unknown config")
    elif task == "text_classification":
        if config == "base":
            model_name = "textattack/bert-base-uncased-SST-2"
            seq_len = 128
            target_microbatch = 128
        elif config == "large":
            model_name = "assemblyai/bert-large-uncased-sst2"
            seq_len = 384
            target_microbatch = 128
        else:
            raise RuntimeError("Unknown config")
    elif task == "keyword_extraction":
        if config == "base":
            model_name = "yanekyuk/bert-uncased-keyword-extractor"
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
        cfg = BertConfig.from_pretrained(model_name)
        model = BertModel(config=cfg)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_bert_na", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyNLPDataset(
            microbatch=microbatch, seq_len=seq_len, hidden_size=cfg.hidden_size, type_ids="token_type_ids"
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "text_classification":

        # Load model and tokenizer
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_bert_text_classification", model)}
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

    elif task == "keyword_extraction":

        # Load model and tokenizer
        model = BertForTokenClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_bert_keyword_extraction", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Load memray/stackexchange Dataset
        memray_dataset = load_dataset("memray/stackexchange", split="validation[:1024]")
        dataset = StackExchangeDataset(dataset=memray_dataset, tokenizer=tokenizer, split=None, seq_len=seq_len)

        # Define evaluation function
        def eval_fn(outputs, labels):
            """Evaluates percentage of samples where 1 or more tokens are correctly highlighted.
            This does not account for:
            1. keywords not in the sample text, 0 score is given for these samples which negatively impacts the mean score.
            2. multiple keywords in the sample text
            """
            batch_eval_scores = []
            for idx in range(len(outputs)):
                if device == "tt":
                    batch_out = outputs[idx][0].value()
                else:
                    batch_out = outputs[idx][0].detach().cpu()
                # accept tokens with > 0 activation
                # model['device'].module.config.id2label
                # {0: 'O', 1: 'B-KEY', 2: 'I-KEY'}
                pred_token_mask = torch.argmax(batch_out, axis=-1) > 0
                true_token_mask = labels[idx].squeeze()
                # remove samples where true label is not in sample tokens
                # these are impossible for token classification to predict
                has_token_mask = true_token_mask.sum(-1) > 0
                true_token_mask = true_token_mask[has_token_mask]
                pred_token_mask = pred_token_mask[has_token_mask]
                # eval: intersection over union of correct keyword tokens highlighted
                batch_interesction = torch.mul(pred_token_mask, true_token_mask).sum(-1)
                batch_union = (true_token_mask + pred_token_mask).sum(-1)
                batch_eval_scores.append((batch_interesction / batch_union).sum())
            # the benchmark script averages the score per sample
            eval_score = torch.stack(batch_eval_scores).sum().item()
            return eval_score

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
