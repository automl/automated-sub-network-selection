import json
import argparse
import os
import pathlib
import torch.nn as nn
import torch

from litgpt import Config
from litgpt.scripts.download import download_from_hub

from whittle.models.gpt import GPT
from whittle.metrics.parameters import compute_parameters
from whittle.modules.linear import Linear
from whittle.modules.embedding import Embedding
from whittle.eval.utils import convert_and_evaluate


def find_layers(module, layers=[Linear, Embedding], name=""):
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


def prune_magnitude(model, prune_n=0, prune_m=0):
    layers = [model.transformer, model.lm_head]

    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            W_mask = torch.zeros_like(W) == 1
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:, ii : (ii + prune_m)].float()
                    W_mask.scatter_(
                        1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True
                    )

            W[W_mask] = 0


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
    checkpoint_dir = f"checkpoints/" + args.model

    # Check if the model checkpoint file exists
    if not os.path.exists(checkpoint_dir):
        download_from_hub(repo_id=args.model)
    config = Config.from_file(checkpoint_dir + "/model_config.yaml")
    config.fix_head_size = True
    config.model_type = "gpt"
    config.tie_embeddings = False
    gpt = GPT(config)
    gpt.name_or_path = checkpoint_dir
    if args.model.startswith("fine-tuned"):
        gpt.load_state_dict(
            torch.load(checkpoint_dir + "/lit_model.pth", map_location="cpu")["model"]
        )
    else:
        gpt.load_state_dict(
            torch.load(checkpoint_dir + "/lit_model.pth", map_location="cpu")
        )
    gpt.reset_super_network()
    params_total = compute_parameters(gpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt.to(torch.bfloat16)
    gpt.to(device)
    #gpt.max_seq_length = 1205
    prune_magnitude(gpt, prune_n=args.n, prune_m=args.m)
    gpt.eval()
    torch.save(
        gpt.state_dict(), f"{checkpoint_dir}/gpt_model_{args.n}_{args.m}_magnitude.pth"
    )
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            torch.cuda.empty_cache()
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
    print(list(results["results"][args.dataset].keys()))
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
        "--dataset",
        type=str,
        default="arc_easy",
        help="Dataset to use for evaluation (default: arc_easy)",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="output dir for final results"
    )

    # Step 2: Parse the arguments
    args = parser.parse_args()
    acc, sparsity_ratio = prune_n_m(args)
    os.makedirs(args.output_dir)
    fh = open(pathlib.Path(args.output_dir) / "results.json", "w")
    results = {
        "accuracy": acc,
        "params": sparsity_ratio,
        "n": args.n,
        "m": args.m,
        "model_type": args.model,
        "task": args.dataset,
        "method": "magnitude_structured",
    }

    json.dump(results, fh)
