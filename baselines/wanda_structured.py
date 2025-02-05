import time
import heapq
import torch
import torch.nn as nn
import transformers
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


def check_sparsity(model):
    use_cache = model.config.use_cache
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


def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.max_seq_length, model.config.n_embd),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
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
                    model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(
        sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1
    )
    W_mask = W_metric <= thres
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_wanda(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibration data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.max_seq_length,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            args, model, dataloader, device
        )

    layers = model.transformer.h
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j], _, _, _, _,_,_ = layer(
                    inps[j].unsqueeze(0),
                    mask=attention_mask,
                    cos=model.cos,
                    sin=model.sin,
                    input_pos=position_ids,
                )  # attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    mask=attention_mask,
                    cos=model.cos,
                    sin=model.sin,
                    input_pos=position_ids,
                )[0]
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
    gpt.name_or_path = "checkpoints/" + args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    prune_wanda(
        args,
        gpt,
        tokenizer,
        prune_n=args.n,
        prune_m=args.m,
    )

    gpt.eval()
    torch.save(
        gpt.state_dict(),
        f"checkpoints/{args.model}/gpt_model_{args.n}_{args.m}_wanda.pth",
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
        "--nsamples", type=int, default=32, help="number of samples for wanda"
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
    print(prune_n_m(args))
