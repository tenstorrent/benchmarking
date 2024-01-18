import torch
from torch.utils.data import DataLoader
from transformers import SqueezeBertConfig, SqueezeBertModel

from ...common import DummyNLPDataset, benchmark_model, torch_df_from_str


@benchmark_model(configs=["base"])
def squeezebert(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

    # Set model parameters based on chosen task and model configuration
    if task == "na":
        if config == "base":
            model_name = "squeezebert/squeezebert-uncased"
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
        cfg = SqueezeBertConfig.from_pretrained(model_name)
        model = SqueezeBertModel(config=cfg)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_squeezebert_na", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyNLPDataset(
            microbatch=microbatch, seq_len=seq_len, hidden_size=cfg.hidden_size, type_ids="token_type_ids"
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
