import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertModel, AlbertTokenizer

from ...common import DummyNLPDataset, SST2Dataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["base", "large", "xlarge"])
def albert(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

        pybuda.config.set_configuration_options(
            default_df_override=pybuda.DataFormat.Float16,
            amp_level=2,
        )

    # Set model parameters based on chosen task and model configuration
    if task == "na":
        if config == "base":
            model_name = "albert-base-v2"
            seq_len = 128
            target_microbatch = 128
        elif config == "large":
            model_name = "albert-large-v2"
            seq_len = 128
            target_microbatch = 128
        elif config == "xlarge":
            model_name = "albert-xlarge-v2"
            seq_len = 128
            target_microbatch = 128
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{8*1024}"
        else:
            raise RuntimeError("Unknown config")
    elif task == "text_classification":
        if config == "base":
            model_name = "textattack/albert-base-v2-imdb"
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
        cfg = AlbertConfig.from_pretrained(model_name)
        model = AlbertModel(config=cfg)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_albert_na", model)}
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
        model = AlbertForSequenceClassification.from_pretrained(model_name)
        tokenizer = AlbertTokenizer.from_pretrained(model_name)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_albert_text_classification", model)}
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
