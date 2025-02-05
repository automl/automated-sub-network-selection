import time
import heapq
import torch
import torch.nn as nn
from baselines.sparsegpt import SparseGPT
from baselines.layerwrapper import WrappedGPT
from baselines.data import get_loaders

from baselines.ablate import AblateGPT
import json
import argparse
import os

import torch.nn as nn
import torch

from litgpt import Config
from litgpt.scripts.download import download_from_hub


from whittle.models.gpt import GPT
from whittle.metrics.parameters import compute_parameters
from whittle.modules.linear import Linear
from whittle.modules.embedding import Embedding
from whittle.eval.utils import convert_and_evaluate
from litgpt import Tokenizer
import transformers


def find_layers(module, layers=[Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev="cuda", prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print("Starting ...")
    args.sparsity_ratio = None
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.max_seq_length,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    # if "model.embed_tokens" in model.hf_device_map:
    #    dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.max_seq_length, model.config.n_embd),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, cos, sin, mask, input_pos):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = mask
            cache["position_ids"] = input_pos
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i]
        # Only for llama
        # if f"model.layers.{i}" in model.hf_device_map:
        #    dev = model.hf_device_map[f"model.layers.{i}"]
        #    print(f"layer {i} device {dev}")
        #    inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    mask=attention_mask,
                    cos=model.cos,
                    sin=model.sin,
                    input_pos=position_ids,
                )[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Pruning ...")

            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                mask=attention_mask,
                cos=model.cos,
                sin=model.sin,
                input_pos=position_ids,
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def compute_sparsity_ratio(gpt, total_params):
    params = 0
    for n, p in gpt.named_parameters():
        params = params + torch.sum(p.data != 0)
    return (params) / total_params


def check_sparsity(model):
    use_cache = False
    model.config.use_cache = False

    layers = model.transformer.h
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prune_n_m(args, shot, metric):
    config_path = f"checkpoints/{args.model}/model_config.yaml"
    model_path = f"checkpoints/{args.model}/lit_model.pth"

    # Check if the model checkpoint file exists
    if not os.path.exists(model_path):
        download_from_hub(repo_id=args.model)
    config = Config.from_file(str("checkpoints/" + args.model + "/model_config.yaml"))
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False
    config.use_cache = True
    gpt = GPT(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt.name_or_path = "checkpoints/" + args.model
    checkpoint_dir = "checkpoints/" + args.model
    gpt.load_state_dict(
        torch.load(
            str("checkpoints/" + args.model + "/lit_model.pth"), map_location="cpu"
        )
    )
    gpt.to(torch.bfloat16)
    gpt.to("cuda")
    gpt.reset_super_network()
    params_total = compute_parameters(gpt)
    tokenizer = transformers.AutoTokenizer.from_pretrained(f"checkpoints/{args.model}/")

    #gpt.max_seq_length = 1205
    prune_sparsegpt(args, gpt, tokenizer, prune_n=args.n, prune_m=args.m, dev=device)
    gpt.eval()
    torch.save(
        gpt.state_dict(), f"{checkpoint_dir}/gpt_model_{args.n}_{args.m}_sparsegpt.pth"
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            torch.cuda.empty_cache()
            #gpt.max_seq_length = 1205
            convert_and_evaluate(
                gpt,
                out_dir="nm/",
                device=None,
                dtype=torch.float32,
                tasks=args.dataset,
                num_fewshot=shot,
                batch_size=args.batch_size,  # Test for non-positive integer
                #limit=1
            )
    with open(str("nm/results.json"), "r") as f:
        results = json.load(f)
    acc = results["results"][args.dataset][metric]
    return acc, compute_sparsity_ratio(gpt, params_total).item()


def get_task_metric_map(dataset):
    if dataset == "winogrande":
        return "acc"
    elif dataset == "arc_challenge":
        return "acc_norm"
    elif dataset == "mmlu":
        return "acc"
    elif dataset == "hellaswag":
        return "acc_norm"
    elif dataset == "gsm8k":
        return "acc"
    elif dataset == "truthfulqa":
        return "mc2"
    else:
        return "acc_norm"


def main():
    parser = argparse.ArgumentParser(
        description="N:M Pruning with specified configuration."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-410m",
        help="Type of model to use",
    )
    parser.add_argument(
        "--n", type=int, default=2, help="Value of n for pruning (default: 2)"
    )
    parser.add_argument(
        "--m", type=int, default=4, help="Value of m for pruning (default: 4)"
    )
    parser.add_argument(
        "--nsamples", type=int, default=4, help="number of samples for wanda"
    )
    parser.add_argument("--seed", type=int, default=9001, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="arc_easy",
        help="Dataset to use for evaluation (default: arc_easy)",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    # Step 2: Parse the arguments
    args = parser.parse_args()
    args.sparsity_ratio = args.n / args.m
    print(prune_n_m(args))
