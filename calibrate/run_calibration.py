import os
import argparse
import pickle
import torch
import json
import transformers
from whittle.models.gpt.model import GPT
from datasets import load_dataset
from litgpt.scripts.download import download_from_hub
from litgpt import Config
from calibrate.utils import (
    get_architecture_grid,
    get_architecture_grid_mag,
    compute_perplexity_wikitext,
    compute_magnitude,
)
from whittle.sampling.random_sampler import RandomSampler
from syne_tune.config_space import randint, lograndint, choice
from whittle.metrics.parameters import (
    compute_all_parameters,
    compute_parameters,
)
from src.utils.sampler import (
    RandomSampler,
    FixGridSampler,
    FixParamGridSampler,
    CalibFixGridSampler,
    ImportanceSampler,
    ImportanceParamGridSampler,
    ImportanceCalibFixGridSampler,
)
from src.utils.search_spaces import SMALL, MEDIUM, HWGPTBench, search_spaces


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = os.path.join(args.chechpoint_dir, args.model_id, "model_config.yaml")
    config_path_hf = os.path.join(args.chechpoint_dir, args.model_id, "config.json")
    model_path = args.path_to_sorted_model

    # Check if the model checkpoint file exists
    # if not os.path.exists(model_path):
    #    download_from_hub(repo_id=args.model_id)

    config = Config.from_file(config_path)
    config.fix_head_size = True
    config.model_type = "gpt"
    with open(config_path_hf) as f:
        hf_config = json.load(f)
    config.tie_embeddings = hf_config["tie_word_embeddings"]

    model = GPT(config)
    search_space = search_spaces[args.search_space](config)
    sampler = ImportanceSampler(
        "/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/importance_orders_llama_joint_mean_block_importance.pkl",
        search_space,
        seed=42,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        f"{args.chechpoint_dir}/{args.model_id}/"
    )

    model.name_or_path = os.path.join(args.chechpoint_dir, args.model_id)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.reset_super_network()
    model.to(torch.bfloat16)
    model.to(device)
    if args.objective == "ppl":
        ppls, params, arch_configs = compute_perplexity_wikitext(
            model,
            search_space,
            sampler,
            tokenizer,
            args.search_space,
            num_archs=args.arch_evals,
            num_bins=args.num_bins,
        )
        architecture_grid = get_architecture_grid(
            sampler, ppls, params, arch_configs, num_bins=args.num_bins
        )
    if args.objective == "mag":
        mags, params, arch_configs = compute_magnitude(
            model,
            search_space,
            sampler,
            tokenizer,
            args.search_space,
            num_archs=args.arch_evals,
            num_bins=args.num_bins,
        )
        architecture_grid = get_architecture_grid_mag(
            sampler, mags, params, arch_configs, num_bins=args.num_bins
        )
    #if args.importance_objective is None:
    with open(
        f"{args.chechpoint_dir}/{args.model_id}/grid_{args.search_space}_{args.objective}.pkl",
        "wb",
    ) as f:
        pickle.dump(architecture_grid, f)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT model calibration.")
    parser.add_argument(
        "--chechpoint_dir",
        type=str,
        default="checkpoints/",
        help="path to litgpt checkpoints",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID to use (e.g., EleutherAI/pythia-410m).",
    )
    parser.add_argument(
        "--search_space", type=str, default="llama_joint", help="search_space_type"
    )
    parser.add_argument(
        "--arch_evals", type=int, default=10, help="number of archs for calibration"
    )
    parser.add_argument("--num_bins", type=int, default=5, help="number of grid points")
    parser.add_argument(
        "--objective", type=str, default="ppl", help="objectives ppl or mag"
    )
    parser.add_argument("--path_to_sorted_model", type=str, default="/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/permuted_model_llama_joint_mean_block_importance.pth")
    args = parser.parse_args()
    main(args)

# test this script using
# python calibrate/run_calibration.py --model_id EleutherAI/pythia-410m --arch_evals 2 --num_bins 22
