import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Define methods, datasets, and base path
base_path = "results-finetune-pythia410m-"
methods = [
    "grid-params",
    "random",
    "importance-random",
    "importance-grid-params",
    "importance-calibrate-mag",
    "standard",
]
datasets = ["winogrande", "arc_easy"]

# Create a directory to store plots if it doesn't exist
output_dir = "pareto_plots_pythia410m"
os.makedirs(output_dir, exist_ok=True)

# Predefine consistent colors for each method
method_colors = {
    "random": "red",
    "grid-params": "blue",
    "importance-calibrate-mag": "green",
    "standard": "orange",
    "importance-random": "purple",
    "importance-calibrate-ppl": "brown",
    "calibrate-mag": "pink",
    "calibrate-ppl": "cyan",
    "importance-grid-params": "yellow",
}

# Predefine unique markers for each method
method_markers = {
    "random": "o",  # Circle
    "grid-params": "s",  # Square
    "importance-calibrate-mag": "^",  # Triangle
    "standard": "D",  # Diamond
    "importance-random": "P",  # Plus-filled
    "importance-calibrate-ppl": "X",  # X-filled
    "calibrate-mag": "*",  # Star
    "calibrate-ppl": "h",  # Hexagon
    "importance-grid-params": "v",  # Downward triangle
}


# Pareto-optimality check function
def is_pareto_optimal(params, accuracy):
    """Compute Pareto-optimal points based on parameters and accuracy values."""
    pareto_mask = np.ones(
        len(params), dtype=bool
    )  # Start with all points being Pareto-optimal
    for i, (p1, a1) in enumerate(zip(params, accuracy)):
        # Any point is not Pareto-optimal if another point has higher accuracy and equal or fewer params
        for j, (p2, a2) in enumerate(zip(params, accuracy)):
            if p2 <= p1 and a2 > a1:  # Dominating condition (maximizing accuracy)
                pareto_mask[i] = False
                break
    return pareto_mask


# Loop over datasets
for dataset in datasets:
    plt.figure(figsize=(10, 6))

    # Loop over methods for each dataset
    for method in methods:
        file_path = f"{base_path}{method}-{dataset}/results.json"
        method_color = method_colors[method]
        method_marker = method_markers[method]

        # Open the file and load the data
        try:
            with open(file_path, "r") as file:
                data = json.load(file)

            params = np.array(data["params"])  # assuming params is a 1D list
            error = np.array(data["error"])  # Read errors from the file
            accuracy = 1 - error  # Convert errors to accuracy (1 - error)

            # Check for Pareto-optimal points dynamically
            pareto_mask = is_pareto_optimal(params, accuracy)

            # Filter Pareto-optimal points using the computed pareto mask
            pareto_params = params[pareto_mask]
            pareto_accuracy = accuracy[pareto_mask]

            # Plot and connect Pareto-optimal points using unique color and marker
            plt.scatter(
                pareto_params,
                pareto_accuracy,
                label=f"{method}",
                s=80,
                edgecolor="black",
                color=method_color,
                marker=method_marker,
            )
            plt.plot(
                pareto_params,
                pareto_accuracy,
                linestyle="-",
                color=method_color,
                marker=method_marker,
                lw=2,
            )

        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping.")
            continue

    # Add labels and title
    plt.xlabel("Fraction of Parameters", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.title(f"Pareto Fronts for {dataset}", fontsize=18, fontweight="bold")

    # Show legend only for Pareto-optimal points
    plt.legend(title="Methods", fontsize=12, title_fontsize=14)

    # Add grid for better readability
    plt.grid(True)
    plt.tight_layout()
    # Save the plot
    plot_file = os.path.join(output_dir, f"pareto_{dataset}.pdf")
    plt.savefig(plot_file)
    plt.close()

    print(f"Pareto front plot saved for {dataset} at {plot_file}")
