import pickle
import numpy as np
import matplotlib.pyplot as plt

# List of methods and datasets
methods = ["random", "importance_calibrate_mag", "standard"]
datasets = ["arc_challenge", "winogrande", "mmlu", "hellaswag", "truthfulqa_mc2"]


# Function to determine Pareto-optimal points
def pareto_optimal(points):
    pareto_points = []
    points_sorted = sorted(
        points, key=lambda x: (x[1], -x[0])
    )  # Sort by params, then by accs in reverse order
    current_best_acc = -np.inf
    for acc, params in points_sorted:
        if acc > current_best_acc:
            pareto_points.append((acc, params))
            current_best_acc = acc
    return np.array(pareto_points)


# Colors for each method
method_colors = {
    "random": "red",
    "grid_params": "blue",
    "random_importance": "purple",
    "importance_grid_params": "yellow",
    "importance_calibrate_mag": "green",
    "standard": "orange",
}

# Markers for each method
method_markers = {
    "random": "o",  # Circle
    "grid_params": "s",  # Square
    "random_importance": "P",  # Plus-filled
    "importance_grid_params": "v",  # Downward triangle
    "importance_calibrate_mag": "^",  # Triangle
    "standard": "D",  # Diamond
}

# Plot each dataset separately
for d in datasets:
    plt.figure(figsize=(10, 7))

    for m in methods:
        # Load the data from the pickle file
        with open(f"grid_eval_{d}_{m}.pkl", "rb") as f:
            data = pickle.load(f)

        accs = np.array(data["accs"])
        params = np.array(data["params"])

        # Get the color and marker for the current method
        color = method_colors[m]
        marker = method_markers[m]

        # Scatter all points using the same color and marker
        plt.scatter(
            params, accs, color=color, marker=marker, alpha=0.5, s=30
        )  # Scatter plot

        # Get Pareto-optimal points
        points = np.array(list(zip(accs, params)))
        pareto_points = pareto_optimal(points)

        # Sort Pareto points by parameter count to create a meaningful line plot
        sorted_indices = np.argsort(pareto_points[:, 1])  # Sort by number of parameters
        pareto_points = pareto_points[sorted_indices]

        # Plot Pareto points as a line with the same color and marker
        plt.plot(
            pareto_points[:, 1],
            pareto_points[:, 0],
            label=m,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=8,
            alpha=0.8,
        )

    # Add titles and labels
    plt.title(f"Pareto Optimal Points for {d}", fontsize=18, fontweight="bold")
    plt.xlabel("Number of Parameters", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.grid(True)  # Add grid
    plt.legend(
        title="Methods", fontsize=12, title_fontsize=14
    )  # Legend for Pareto-optimal lines only

    # Apply tight layout
    plt.tight_layout()

    # Save the plot as a PDF image
    plt.savefig(f"pareto_plot_{d}_llama_3.1_8b.pdf")
    plt.close()
