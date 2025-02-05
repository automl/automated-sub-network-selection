import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the data
with open(
    "/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/grid_eval_hellaswag_galore.pkl",
    "rb",
) as f:
    galore = pickle.load(f)

with open(
    "/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/grid_eval_hellaswag_last_layer.pkl",
    "rb",
) as f:
    last_ft = pickle.load(f)

with open(
    "/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/result_pickle/grid_eval_hellaswag_importance_calibrate_mag.pkl",
    "rb",
) as f:
    lora = pickle.load(f)


# Helper function to compute Pareto front
def pareto_front(params, accs):
    sorted_indices = np.argsort(params)
    sorted_params = params[sorted_indices]
    sorted_accs = accs[sorted_indices]

    pareto_params = []
    pareto_accs = []

    current_max_acc = -np.inf

    # Iterate through sorted points to filter Pareto front
    for p, a in zip(sorted_params, sorted_accs):
        if a > current_max_acc:
            pareto_params.append(p)
            pareto_accs.append(a)
            current_max_acc = a

    return np.array(pareto_params), np.array(pareto_accs)


# Extract accuracies and parameters
galore_accs = np.array(galore["accs"])
last_ft_accs = np.array(last_ft["accs"])
galore_params = np.array(galore["params"]) / max(np.array(lora["params"]))
last_ft_params = np.array(last_ft["params"]) / max(np.array(last_ft["params"]))
lora_accs = np.array(lora["accs"])
lora_params = np.array(lora["params"]) / max(np.array(lora["params"]))

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot for all points without labels
plt.scatter(
    galore_params,
    galore_accs,
    color="blue",
    marker="o",
    s=60,
    alpha=0.5,
    edgecolors="k",
)
plt.scatter(
    last_ft_params,
    last_ft_accs,
    color="red",
    marker="s",
    s=60,
    alpha=0.5,
    edgecolors="k",
)
plt.scatter(
    lora_params, lora_accs, color="green", marker="^", s=60, alpha=0.5, edgecolors="k"
)

# Plot Pareto front for each with labels
galore_pf_params, galore_pf_accs = pareto_front(galore_params, galore_accs)
plt.plot(
    galore_pf_params,
    galore_pf_accs,
    color="blue",
    marker="o",
    linestyle="-",
    linewidth=2,
    markersize=8,
    label="Galore Pareto",
)

last_ft_pf_params, last_ft_pf_accs = pareto_front(last_ft_params, last_ft_accs)
plt.plot(
    last_ft_pf_params,
    last_ft_pf_accs,
    color="red",
    marker="s",
    linestyle="-",
    linewidth=2,
    markersize=8,
    label="Last Layer FT Pareto",
)

lora_pf_params, lora_pf_accs = pareto_front(lora_params, lora_accs)
plt.plot(
    lora_pf_params,
    lora_pf_accs,
    color="green",
    marker="^",
    linestyle="-",
    linewidth=2,
    markersize=8,
    label="LoRA Pareto",
)

# Customize plot
plt.title(
    "Pareto Front Comparison: Last Layer FT, Galore, and LoRA on hellaswag", fontsize=16
)
plt.xlabel("Fraction of Parameters", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend(fontsize=12)  # Only the Pareto fronts will be in the legend
plt.tight_layout()

# Show the plot
plt.savefig("all_ft_hellaswag.pdf")
