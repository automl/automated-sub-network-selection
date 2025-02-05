#!/bin/bash -l

#SBATCH --time=1-00:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16 # number of processor cores (i.e. threads)
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB   # memory per CPU core
#SBATCH -p mldlc2_gpu-l40s
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out        # output file in logs directory
#SBATCH --error=logs/%x_%j.err         # error file in logs directory

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
