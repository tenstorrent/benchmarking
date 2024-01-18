import os

import torch
import torch.multiprocessing
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from ...common import DummyCVDataset, ImageNetDataset, benchmark_model, torch_df_from_str

torch.multiprocessing.set_sharing_strategy("file_system")


@benchmark_model(configs=["121", "169", "201", "161"])
def densenet(training: bool, task: str, config: str, microbatch: int, device: str, data_type: str):

    if device == "tt":
        import pybuda

        compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
        compiler_cfg.balancer_policy = "CNN"
        compiler_cfg.enable_t_streaming = True

    # Set model parameters based on chosen task and model configuration
    img_res = 224
    target_microbatch = 32

    if config == "121":
        model_name = "densenet121"
        compiler_cfg.enable_enumerate_u_kt = False
        os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"

    elif config == "161":
        model_name = "densenet161"
        if device == "tt":
            compiler_cfg.enable_auto_fusing = False
            compiler_cfg.place_on_new_epoch("concatenate_131.dc.sparse_matmul.7.lc2")
            os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
            os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    elif config == "169":
        model_name = "densenet169"
        if device == "tt":
            os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
            os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    elif config == "201":
        model_name = "densenet201"
        if device == "tt":
            os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
            os.environ["PYBUDA_PAD_SPARSE_MM"] = "{11:12}"
            os.environ["PYBUDA_GRAPHSOLVER_SELF_CUT_TYPE"] = "ConsumerOperandDataEdgesFirst"

    else:
        raise RuntimeError("Unknown config")

    # Configure microbatch, if none provided
    if microbatch == 0:
        microbatch = target_microbatch

    # Task specific configuration
    if task == "na":

        # Load model
        model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)

        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_densenet{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        dataset = DummyCVDataset(microbatch=microbatch, channels=3, height=img_res, width=img_res, data_type=data_type)

        # Define evaluation function
        def eval_fn(**kwargs):
            return 0.0

    elif task == "image_classification":

        # Load model
        model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=True)

        version = "torch"
        feature_exctractor = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # Configure model mode for training or evaluation
        if training:
            model.train()
        else:
            model.eval()

        # Create model device placement map
        if device == "tt":
            model = {"device": pybuda.PyTorchModule(f"pt_densenet{config}", model)}
        else:
            model = model.to(device, dtype=torch_df_from_str(data_type))

        # Create random inputs and targets
        imagenet_dataset = load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=True)
        dataset_iter = iter(imagenet_dataset)
        dataset = []
        for _ in range(1000):
            dataset.append(next(dataset_iter))
        dataset = ImageNetDataset(dataset=dataset, feature_extractor=feature_exctractor, version=version)

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
                    output = output.detach().cpu()
                pred_labels.extend(torch.argmax(output, axis=-1))
            for label in labels:
                true_labels.extend(label)
            eval_score = accuracy_metric.compute(references=true_labels, predictions=pred_labels)

            return eval_score["accuracy"]

    else:
        raise RuntimeError("Unknown task")

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True)

    # Add loss function, if training
    if training and device == "tt":
        model["cpu-loss"] = pybuda.PyTorchModule("l1loss", torch.nn.L1Loss())

    return model, generator, eval_fn
