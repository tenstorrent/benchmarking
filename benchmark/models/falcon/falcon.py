# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, FalconForCausalLM, pipeline

from benchmark.common.benchmark_run import OutputType

from ...common import BenchmarkRun, DummyPipelineDataset, PipelineDataset, benchmark_model, torch_df_from_str

# Set random seed for reproducibility
torch.manual_seed(42)


@benchmark_model(configs=["7b", "7b-instruct"])
def falcon(
    training: bool, task: str, config: str, microbatch: int, device: str, data_type: str, benchmark_run: BenchmarkRun
):

    # Set model parameters based on chosen task and model configuration
    if task in ["na", "hellaswag", "text_summarization", "alpacaeval"]:
        if config == "7b":
            model_name = "tiiuae/falcon-7b"
        elif config == "7b-instruct":
            model_name = "tiiuae/falcon-7b-instruct"
        else:
            raise RuntimeError("Unknown config")
    else:
        raise RuntimeError("Unknown task")

    # Set model parameters based on chosen task
    # max_tokens: the maximum number of tokens (prompt + generation) per user row
    # max_new_tokens: the maximum number of tokens to generate per user row
    # defaults
    max_tokens = 1024
    max_new_tokens = 256
    min_new_tokens = 0
    top_p_enable = 1
    if task == "na":
        benchmark_run.output_type = OutputType.TEXT
        top_p_enable = 0
    elif task == "hellaswag":
        max_new_tokens = 1
        min_new_tokens = 1
        benchmark_run.output_type = OutputType.LOGITS
    elif task == "text_summarization":
        benchmark_run.output_type = OutputType.TEXT
    elif task == "alpacaeval":
        benchmark_run.output_type = OutputType.TEXT

    # Configure microbatch, if none provided
    microbatch = 1 if microbatch == 0 else microbatch

    # Create model device placement map
    if device == "tt":
        from benchmark.models.falcon.utils.model import Falcon

        # use all 32, implementation has this shape fixed
        assert microbatch == 32, "microbatch must be 32 for TT, extra rows will be generated with padding"
        model = Falcon(
            user_rows=32,
            num_tokens=max_tokens,
            num_gen_tokens=max_new_tokens,
            top_p_enable=top_p_enable,
            top_p=0.9,
            top_k=40,
            temperature=1.0,
            model_ckpt=model_name,
            output_type=benchmark_run.output_type,
            tti_save=benchmark_run.save_tti,
            tti_load=benchmark_run.load_tti,
        )
        model.initialize()
        tokenizer = model.tokenizer
        benchmark_run.tokenizer = tokenizer
        # using fixed 32 user rows means generaration occurs until all 32 rows are complete
        # this gives a measure of the maximum capacity of the model implementation
        benchmark_run.token_count_use_max_batch = True
        # using prefill via decode on TT device
        benchmark_run.count_prompt_tokens = True

    else:
        device = 0 if device == "cuda" else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        benchmark_run.tokenizer = tokenizer
        if task == "hellaswag":
            # in order to get logits or loss we use the CausalLM implementation
            falcon_model = FalconForCausalLM.from_pretrained(model_name)

            def model_wrapper(batch):
                # this wrapper is necessary because the batch inputs have different token lengths
                # the CausalLM implementation does not accept lists of input_ids
                # batching only works with a tensor (batch_size, input_ids_length)
                # this would be possible with attention_mask usage, however the evaluation
                # indexes would no longer be the same as the TT implementation
                outputs = []
                for prompt in batch:
                    input_ids = tokenizer(prompt, padding=True, return_tensors="pt").input_ids
                    outputs.append(falcon_model(input_ids, labels=input_ids).logits.squeeze(0))
                return outputs

            model = model_wrapper
        else:
            model = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch_df_from_str(data_type),
                return_dict=True,
                output_hidden_states=True,
            )
            # set model parameters
            model.model.config.early_stopping = True
            model.model.config.length_penalty = 1.0
            # disable ngram repeat restriction to not disadvantage other implementations
            model.model.config.no_repeat_ngram_size = 0
            model.model.config.max_length = max_new_tokens
            model.model.config.min_length = min_new_tokens
            model.model.config.num_beams = 1
            model.model.config.num_return_sequences = 1
            # set key for accessing output text
            model.output_key = "generated_text"

    # Task specific configuration
    if task == "na":

        # Create random inputs and targets
        dataset = DummyPipelineDataset(
            microbatch=microbatch,
            sample_text="translate the following sentence from English to German: The house is wonderful.",
            answer="Das Haus ist wunderbar.",
        )

        # Define evaluation function
        def eval_fn(outputs, labels):
            # this tests that all 32 batch outputs are the same when top_p_enable=0
            for i in range(microbatch):
                print(i)
                print(outputs[0][i])

            first_out = outputs[0][0]
            eval_score = float(all([out == first_out for out in outputs[0]]))
            return eval_score

    elif task == "hellaswag":
        """
        HellaSwag dataset: https://arxiv.org/pdf/1905.07830.pdf
        Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations
        Evaluation note:
        Hellawag is a multiple choice task, where the model is given a context and a potential ending.
        The model is then asked to choose the correct ending from a set of 4 options (A,B,C,D).
        The model is scored based on accuracy, i.e. the ratio of correct predictions / total questions.
        Using https://github.com/EleutherAI/lm-evaluation-harness as a popular reference implementation
        Q:= query (aka prompt, or context)
        A:= answer from model (aka completion, or generated text)
        there are several possible ways of obtaining the model "selection" of the 4 completion options:
          1. The training loss function calculated on the 4 Q+A options, select the minimum mean loss.
          2. The next-token prediction calculated on the 4 Q+A options, select the maximum confidence score.
            confidence score is the mean softmax of the logits indexed by the next-token in A.
          3. Use few-shot prompting to get model to explicitly select (A,B,C,D) in multiple choice task.
          4. Compare closeness of full actual A to Q with options (A,B,C,D) and select closest.

        Implementation 1. is not possible without the loss function, for this reason we use option 2.
        because it is the easiest to implement and is the most similar to 1.
        """
        # note: generator.drop_last is set to True, so the last batch will be dropped if it is not full
        hellaswag_dataset = load_dataset("hellaswag", split="validation[:64]")
        dataset = PipelineDataset(
            dataset=hellaswag_dataset,
            input_text="ctx",
            label="label",
            prepend_text="activity_label",
            serialize_hellaswag=True,
            tokenizer=tokenizer,
        )

        # Define evaluation function
        def eval_fn(outputs, labels):
            pred_labels = []
            true_labels = []
            # get option selection for each query, mapped by
            conf_dict = defaultdict(list)
            labels_dict = {}
            for b_out, b_lbl in zip(outputs, labels):
                for out, lbl in zip(b_out, b_lbl):
                    if isinstance(out, list):
                        out = torch.stack(out)
                    # an offset is needed to compare the predicted token with the label token
                    offset = 1
                    start_idx = len(lbl["prompt_ids"]) - offset
                    end_idx = start_idx + len(lbl["end_ids"])
                    if end_idx > (len(out) - offset):
                        # NOTE: an ending or prompt may be tokenized to fewer tokens when tokenized
                        # together than when tokenized separately due to spacing handling. For example:
                        # >>> tokenizer.decode([18943], clean_spaces=True)
                        #    ' demonstrates'
                        # >>> tokenizer.decode([10985,   245,  2266,   750,], clean_spaces=True)
                        #    'demonstrates'
                        # this affects all hardware implmentations the same so it is not a problem for relative comparison
                        # another possible implementation option would be to add leading or trailing space
                        # and remove any space tokens that were added to the output, explictily checking for
                        # this edge case on both LH and RH sides of prompt / ending
                        end_idx = len(out) - offset
                        start_idx = end_idx - len(lbl["end_ids"])
                    eval_out = out.squeeze(0)[start_idx:end_idx, :]
                    conf_dict[lbl["ind"]].append(
                        torch.gather(
                            torch.nn.functional.softmax(eval_out, dim=-1),
                            1,
                            lbl["end_ids"].unsqueeze(1),
                        ).mean()
                    )
                    labels_dict[lbl["ind"]] = int(lbl["label"])

            for ind, conf_list in conf_dict.items():
                pred_labels.append(torch.stack(conf_list).argmax().item())
                true_labels.append(labels_dict[ind])

            accuracy_metric = evaluate.load("accuracy")
            calc_metric = accuracy_metric.compute(references=true_labels, predictions=pred_labels)
            eval_score = calc_metric["accuracy"]
            return eval_score

    elif task == "text_summarization":

        # Load CNN / DailyMail dataset
        cnndm_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:32]")
        dataset = PipelineDataset(
            dataset=cnndm_dataset,
            input_text="article",
            label="highlights",
            prepend_text="summarize: ",
        )

        # Define evaluation function
        def eval_fn(outputs, labels):
            rouge_metric = evaluate.load("rouge")
            pred_labels = []
            true_labels = []
            for batch in outputs:
                for item in batch:
                    if isinstance(item[0], dict):
                        output = item[0]["generated_text"]
                    else:
                        output = item
                    pred_labels.extend([output])
            for label in labels:
                true_labels.extend(label)

            for idx, (pred, input) in enumerate(zip(pred_labels, cnndm_dataset)):
                prompt_len = len(
                    tokenizer("summarize: " + input["article"], return_tensors="pt")["input_ids"].squeeze()
                )
                pred_ids = tokenizer(pred, return_tensors="pt")["input_ids"].squeeze()
                pred_only = pred_ids[prompt_len:]
                output = tokenizer.decode(pred_only).lstrip()
                pred_labels[idx] = output

            eval_score = rouge_metric.compute(references=true_labels, predictions=pred_labels)
            print(eval_score)

            return eval_score["rouge1"]

    elif task == "alpacaeval":
        # NOTE: requires alpaca-eval==0.3.0 and Python>=3.10 for scoring

        # AlpacaEval dataset
        eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval[:128]")
        eval_set = list(eval_set)
        dataset = PipelineDataset(dataset=eval_set, input_text="instruction", label="output")
        # add prompt token lengths for removal from stats
        benchmark_run.prompt_tokens_lens = [len(tokenizer(item[0]).input_ids) for item in dataset]

        # Define evaluation function
        def eval_fn(outputs, labels):
            import json

            pred_labels = []
            true_labels = []
            for batch in outputs:
                for item in batch:
                    if isinstance(item[0], dict):
                        output = item[0]["generated_text"]
                    else:
                        output = item
                    pred_labels.extend([output])
            for label in labels:
                true_labels.extend(label)

            outputs = []
            num_tokens = max_new_tokens
            for pred, input in zip(pred_labels, eval_set):
                prompt_len = len(tokenizer(input["instruction"], return_tensors="pt")["input_ids"].squeeze())
                pred_ids = tokenizer(pred, return_tensors="pt")["input_ids"].squeeze()
                pred_only = pred_ids[prompt_len:]
                output = tokenizer.decode(pred_only).lstrip()

                test_case = (
                    f"falcon-7b-instruct-wh-{num_tokens}tk"
                    if device == "tt"
                    else f"falcon-7b-instruct-a2-{num_tokens}tk"
                )

                outputs.append(
                    {
                        "instruction": input["instruction"],
                        "output": output,
                        "generator": test_case,
                        "dataset": input["dataset"],
                    }
                )

            # Writing to sample.json
            json_object = json.dumps(outputs, indent=4)
            with open(f"benchmark/models/falcon/results/output_{test_case}.json", "w") as outfile:
                outfile.write(json_object)

            return 0.0

    def collate_fn(batch):
        # Separate inputs and labels
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return inputs, labels

    # Create DataLoader
    generator = DataLoader(dataset, batch_size=microbatch, shuffle=False, drop_last=True, collate_fn=collate_fn)

    return model, generator, eval_fn
