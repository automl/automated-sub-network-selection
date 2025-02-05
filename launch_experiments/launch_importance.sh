#!/bin/bash

# Define arrays for search spaces, schemes, and layer schemes
search_spaces=("llama_heads" "llama_head_size" "llama_joint") # "llama_heads" "llama_head_size" "llama_joint"
schemes=("mean" "mean-norm" "norm" "norm-mean" "variance" "variance-mean" "variance-norm" "norm-variance" "mean-variance")
layer_schemes=("perplexity" "block_importance")

# Loop through each combination of search space, scheme, and layer scheme
for space in "${search_spaces[@]}"; do
  for scheme in "${schemes[@]}"; do
    for layer_scheme in "${layer_schemes[@]}"; do
      echo "Submitting job for space: $space, scheme: $scheme, layer_scheme: $layer_scheme"
      sbatch launch_experiments/submit_job.sh "$space" "$scheme" "$layer_scheme"
    done
  done
done