import os
import json
import argparse
import logging

from collections import defaultdict
from pathlib import Path

import pandas as pd

from llm_compression.experiment_configs import ExperimentConfig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--search_space", type=str, default=None)

    args, _ = parser.parse_known_args()

    base_path = Path.home() / "experiments" / "llm_compression" / "evaluation_results"

    nas_keys = ExperimentConfig.keys()
    all_results = defaultdict(list)
    for root, dirs, files in os.walk(base_path):
        for file in files:
            fname = Path(root) / file
            try:
                results = json.load(open(fname))
            except json.decoder.JSONDecodeError:
                continue
            all_results["accuracy"].append(results["accuracy"])
            all_results["params"].append(results["params"])
            for key in nas_keys:
                if key in results:
                    if key == "model_type":
                        all_results[key].append(results[key].replace("/", "-"))
                    else:
                        all_results[key].append(results[key])
                else:
                    all_results[key].append(None)

all_results = pd.DataFrame(all_results)
all_results.to_csv("results_evaluation.csv")
