#!/bin/bash -l

#SBATCH --time=1-00:00:00   # Walltime (1 day)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16  # Number of CPU cores
#SBATCH --ntasks-per-node=1 # Single task on the node
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --mem=100GB         # Total memory
#SBATCH -p alldlc2_gpu-l40s  # Partition
#SBATCH --exclusive         # Exclusive node usage
#SBATCH --requeue           # Requeue if preempted
#SBATCH --output=logs/%x_%j.out  # Output log
#SBATCH --error=logs/%x_%j.err   # Error log

# Set variables from command line arguments
space=$1
scheme=$2
layer_scheme=$3

export WANDB_MODE=offline
export PYTHONPATH=.

echo "Running with space: $space, scheme: $scheme, layer_scheme: $layer_scheme"

python src/compute_importance.py \
  --model_id meta-llama/Meta-Llama-3.1-8B \
  --num_batches 100 \
  --space "$space" \
  --objective "$scheme" \
  --batch_size 4 \
  --max_seq_len 1024 \
  --n_configs 500 \
  --layer_scheme "$layer_scheme"

