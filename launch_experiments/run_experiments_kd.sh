#!/bin/bash

# Define the lists of possible values for each parameter
kd_loss_values=("forward_kl" "js_distance")
kd_alpha_values=(0.5)
kd_beta_values=(2)
kd_temperature_values=(1.5 0.8 0.9)   # Add kd_temperature values
weight_scheme_values=("custom")
weight_supernet_loss_values=(false)  # Boolean values
dataset_names=("commonsense") # "math")
sampling_strategies=("importance-calibrate") # "importance-grid-params")
train_strategies=("sandwich-kd")

# Iterate over all possible combinations
for kd_loss in "${kd_loss_values[@]}"; do
  for kd_alpha in "${kd_alpha_values[@]}"; do
    for kd_beta in "${kd_beta_values[@]}"; do
      for kd_temperature in "${kd_temperature_values[@]}"; do  # Add new loop for kd_temperature
        for weight_scheme in "${weight_scheme_values[@]}"; do
          for weight_supernet_loss in "${weight_supernet_loss_values[@]}"; do
            for dataset in "${dataset_names[@]}"; do
              for sampling_strategy in "${sampling_strategies[@]}"; do
                for train_strategy in "${train_strategies[@]}"; do
                  
                  # Submit each job to sbatch, passing the parameters to the job script
                  sbatch job_scripts/submit_small.sh "$kd_loss" "$kd_alpha" "$kd_beta" "$kd_temperature" "$weight_scheme" "$weight_supernet_loss" "$dataset" "$sampling_strategy" "$train_strategy"
                  
                done
              done
            done
          done
        done
      done
    done
  done
done

