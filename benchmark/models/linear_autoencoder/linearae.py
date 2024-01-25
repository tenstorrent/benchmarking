# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.data import DataLoader

from ...common import DummyCVDataset_1D, benchmark_model, torch_df_from_str


class LinearAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder_lin1 = torch.nn.Linear(784, 128)
        self.encoder_lin2 = torch.nn.Linear(128, 64)
        self.encoder_lin3 = torch.nn.Linear(64, 12)
        self.encoder_lin4 = torch.nn.Linear(12, 3)

        # Decoder
        self.decoder_lin1 = torch.nn.Linear(3, 12)
        self.decoder_lin2 = torch.nn.Linear(12, 64)
        self.decoder_lin3 = torch.nn.Linear(64, 128)
        self.decoder_lin4 = torch.nn.Linear(128, 784)

        # Activation Function
        self.act_fun = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        act = self.encoder_lin1(x)
        act = self.act_fun(act)
        act = self.encoder_lin2(act)
        act = self.act_fun(act)
        act = self.encoder_lin3(act)
        act = self.act_fun(act)
        act = self.encoder_lin4(act)

        # Decode
        act = self.decoder_lin1(act)
        act = self.act_fun(act)
        act = self.decoder_lin2(act)
        act = self.act_fun(act)
        act = self.decoder_lin3(act)
        act = self.act_fun(act)
        act = self.decoder_lin4(act)

        return act


@benchmark_model(configs=["1"])
def linear_ae(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

        compiler_cfg = pybuda.config._get_global_compiler_config()
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True

    # Set model parameters based on chosen task and model configuration
    img_res = 28
    target_microbatch = 32

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Load model
        model = LinearAE()

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule("pt_linear_ae", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset_1D(
            microbatch=microbatch, channels=1, height=img_res, width=img_res, data_type=data_type
        )

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
