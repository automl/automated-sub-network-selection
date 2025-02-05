import os
import argparse
import pickle
from baselines.magnitude_structured import prune_n_m as magnitude
from baselines.wanda_structured import prune_n_m as wanda
from baselines.prune_sparsegpt import prune_n_m as sparsegpt


def load_existing_results(output_dir):
    """Load existing results from a pickle file if it exists."""
    results_file = os.path.join(output_dir, "results.pkl")
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    return None


def save_results(errors, params, latencies, processed_nms, method, args):
    """Save the current results to a pickle file."""
    results = {
        "dataset": args.dataset,
        "configs": processed_nms,  # Only save N:M pairs that have been processed
        "accuracy": errors,
        "params": params,
        "latencies": latencies,  # Save latencies
        "args": {
            "model_type": "baselines-" + args.model.replace("/", "-") + "-" + method,
            "task": args.dataset,
            "search_space": "hw_gpt_bench",
            "search_strategy": "grid",
            "seed": 1,
            "objective": "acc",
            "iterations": 100,
            "method": method,
        },
        "is_pareto_optimal": [True for _ in range(len(errors))],
    }

    output_dir = f"baselines-{args.model.replace('/', '-')}-{method}-{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "results.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {output_file}")


def results_exist_for_grid_point(existing_results, nm):
    """Check if results for a specific N:M grid point already exist in the loaded results."""
    if existing_results is None:
        return False

    # If the accuracy for a specific N:M pair is already recorded, we assume it's done
    n_m_pair = (nm["n"], nm["m"])
    for config in existing_results["configs"]:
        if config["n"] == n_m_pair[0] and config["m"] == n_m_pair[1]:
            return True

    return False


def process_nm_grid_point(
    method_name,
    prune_method,
    nm,
    errors,
    sparsities,
    latencies,
    processed_nms,
    existing_results,
    args,
    shot,
    metric
):
    """Process a single N:M grid point for a given method."""
    if results_exist_for_grid_point(existing_results, nm):
        print(
            f"Skipping {method_name} for N:M = {nm['n']}:{nm['m']} (already processed)"
        )
        return

    args.n = nm["n"]
    args.m = nm["m"]
    print(
        f"Pruning with {method_name} method for N:M = {args.n}:{args.m} on dataset {args.dataset}"
    )

    # Call the pruning method (magnitude, wanda, or sparsegpt)
    acc, sparsity = prune_method(args, shot, metric)

    # Append the results for this N:M configuration
    errors.append(acc)
    sparsities.append(sparsity)
    processed_nms.append(nm)

    # Save the updated results
    save_results(errors, sparsities, latencies, processed_nms, method_name, args)


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
        "--nsamples", type=int, default=32, help="Number of samples for wanda"
    )
    parser.add_argument("--seed", type=int, default=9001, help="Random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sciq",
        help="Dataset to use for evaluation (default: arc_easy)",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")

    # Parse the arguments
    args = parser.parse_args()

    # N:M grid
    nm_list = [
        {"n": 2, "m": 4},  # 0.627480
    ]

    # Datasets to iterate over
    task_to_shot = {
        "winogrande":5,
        "hellaswag":10,
        "arc_challenge":25,
        "arc_easy":0,
        "piqa":0,
        "boolq":0,
        "truthfulqa_mc2":0,
        "gsm8k":5,
        "mmlu":5,
        "lambada_openai":0,
        #"minerva_math":4,
        "mathqa":0
    }
    metric_map = {
        "winogrande" :"acc,none", 
        "arc_challenge":"acc_norm,none",
        "arc_easy":"acc_norm,none",
        "mmlu":"acc,none",
        "hellaswag":"acc_norm,none",
        "gsm8k":"exact_match,strict-match",
        "truthfulqa_mc2": "acc,none",
        "boolq":"acc,none",
        "piqa":"acc,none",
        "lambada_openai":"acc,none",
        #"minerva_math":"exact_match,none",
        "mathqa":"acc_norm,none"
    }
    datasets = list(metric_map.keys())

    # Iterate over datasets
    for dataset in datasets:
        args.dataset = dataset
        print(f"Processing dataset: {dataset}")
        n_shot = task_to_shot[dataset]
        metric = metric_map[dataset]
        # Directory names for each method and dataset
        output_dir_mag = (
            f"baselines-{args.model.replace('/', '-')}-magnitude-nm-{args.dataset}"
        )
        output_dir_wanda = (
            f"baselines-{args.model.replace('/', '-')}-wanda-nm-{args.dataset}"
        )
        output_dir_sparsegpt = (
            f"baselines-{args.model.replace('/', '-')}-sparsegpt-nm-{args.dataset}"
        )

        # Load existing results to resume if possible
        existing_results_mag = load_existing_results(output_dir_mag)
        existing_results_wanda = load_existing_results(output_dir_wanda)
        existing_results_sparsegpt = load_existing_results(output_dir_sparsegpt)

        # Initialize results containers, resume from existing results if available
        errors_mag = existing_results_mag["accuracy"] if existing_results_mag else []
        sparsities_mag = existing_results_mag["params"] if existing_results_mag else []
        latencies_mag = (
            existing_results_mag["latencies"] if existing_results_mag else []
        )
        processed_nms_mag = (
            existing_results_mag["configs"] if existing_results_mag else []
        )

        errors_wanda = (
            existing_results_wanda["accuracy"] if existing_results_wanda else []
        )
        sparsities_wanda = (
            existing_results_wanda["params"] if existing_results_wanda else []
        )
        latencies_wanda = (
            existing_results_wanda["latencies"] if existing_results_wanda else []
        )
        processed_nms_wanda = (
            existing_results_wanda["configs"] if existing_results_wanda else []
        )

        errors_sparsegpt = (
            existing_results_sparsegpt["accuracy"] if existing_results_sparsegpt else []
        )
        sparsities_sparsegpt = (
            existing_results_sparsegpt["params"] if existing_results_sparsegpt else []
        )
        latencies_sparsegpt = (
            existing_results_sparsegpt["latencies"]
            if existing_results_sparsegpt
            else []
        )
        processed_nms_sparsegpt = (
            existing_results_sparsegpt["configs"] if existing_results_sparsegpt else []
        )

        # Process each N:M grid point
        for nm in nm_list:
            # Process magnitude pruning
            process_nm_grid_point(
                "magnitude-nm",
                magnitude,
                nm,
                errors_mag,
                sparsities_mag,
                latencies_mag,
                processed_nms_mag,
                existing_results_mag,
                args,
                n_shot,
                metric
            )

            # Process wanda pruning
            process_nm_grid_point(
                "wanda-nm",
                wanda,
                nm,
                errors_wanda,
                sparsities_wanda,
                latencies_wanda,
                processed_nms_wanda,
                existing_results_wanda,
                args,
                n_shot,
                metric
            )

            # Process sparsegpt pruning
            process_nm_grid_point(
                "sparsegpt-nm",
                sparsegpt,
                nm,
                errors_sparsegpt,
                sparsities_sparsegpt,
                latencies_sparsegpt,
                processed_nms_sparsegpt,
                existing_results_sparsegpt,
                args,
                n_shot,
                metric
            )


if __name__ == "__main__":
    main()
