#!/bin/bash -l

#SBATCH --time=1-0:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB   # memory per CPU core
#SBATCH -p accelerated
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out        # output file in logs directory
#SBATCH --error=logs/%x_%j.err         # error file in logs directory

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export WANDB_MODE=offline
export PYTHONPATH=.
kd_loss="$1"
kd_alpha="$2"
kd_beta="$3"
kd_temperature="$4"
weight_scheme="$5"
weight_supernet_loss="$6"
dataset="$7"
sampling_strategy="$8"
train_strategy="$9"

# Set dataset-specific arguments
data_args=""
epochs=10
if [ "$dataset" == "commonsense" ]; then
  data_args="--data JSON --data.json_path commonsense_170k.json --data.val_split_fraction 0.1"
  epochs=5
elif [ "$dataset" == "math" ]; then
  data_args="--data JSON --data.json_path math10k.json --data.val_split_fraction 0.1"
  epochs=10
fi

# Run the Python script with the given parameters and conditional data arguments
python src/finetuning/lora.py meta-llama/Meta-Llama-3.1-8B \
  --config config_hub/finetune/llama-3.1-8b/lora.yaml \
  --sampling_strategy "$sampling_strategy" \
  --out_dir test_lora/ \
  --search_space llama2 \
  --train.epochs $epochs \
  --kd_loss "$kd_loss" \
  --kd_temperature "$kd_temperature" \
  --kd_alpha "$kd_alpha" \
  --kd_beta "$kd_beta" \
  --weight_scheme "$weight_scheme" \
  --weight_supernet_loss "$weight_supernet_loss" \
  --dataset "$dataset" \
  --train_strategy "$train_strategy" \
  $data_args  # Add data-specific arguments if set
