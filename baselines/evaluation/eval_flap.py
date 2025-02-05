import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.hf_llama.modeling_llama import LlamaForCausalLM

from importlib.metadata import version

from lib.prune import prune_flap, check_sparsity
from lib.eval import eval_ppl
import lm_eval
from lm_eval.models.huggingface import HFLM
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", type=str, default="meta-llama/Meta-Llama-3-8B", help="LLaMA model"
)  # Huggingface model name
parser.add_argument(
    "--seed", type=int, default=0, help="Seed for sampling the calibration data."
)
parser.add_argument(
    "--nsamples", type=int, default=1024, help="Number of calibration samples."
)
parser.add_argument("--pruning_ratio", type=float, default=0.5, help="Pruning ratio.")
parser.add_argument("--remove_heads", type=int, default=-1, help="Remove num_heads")
parser.add_argument(
    "--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", "N/A"]
)
parser.add_argument("--structure", type=str, default="AL-AM", choices=["AL-AM"])
parser.add_argument("--prune_method", type=str, default="flap", choices=["flap"])
parser.add_argument("--cache_dir", default="llm_weights", type=str)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--save_model", type=str, default=None, help="Path to save the pruned model."
)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument(
    "--gqa_groups", type=int, default=4, help="Number of gqa groups, 1 for no GQA."
)
parser.add_argument(
    "--start_pruning_layer_idx",
    type=int,
    default=22,
    help="Layer idx post which pruning starts",
)
parser.add_argument("--head_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=4096)
args = parser.parse_args()


def get_llm(model, device):
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, device_map=device, trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float16, trust_remote_code=True
        ).to(torch.device(f"cuda:{device}"))

    model.seqlen = 128
    return model


model = get_llm(args.model, args.device)
lm_obj = HFLM(pretrained=model)
task_manager = lm_eval.tasks.TaskManager()
results = []
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["winogrande"],
    num_fewshot=5,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["gsm8k"],
    num_fewshot=5,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["mmlu"],
    num_fewshot=5,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["boolq"],
    num_fewshot=0,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["arc_challenge"],
    num_fewshot=25,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
result = lm_eval.simple_evaluate(
    model=lm_obj,
    tasks=["hellaswag"],
    num_fewshot=10,
    task_manager=task_manager,
    batch_size="auto",
)
results.append(result["results"])
print(results)
