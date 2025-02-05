import pickle
import numpy as np
import matplotlib.pyplot as plt


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


# Load data from pickle files
with open("grid_eval_arc_easy_importance_calibrate_mag_pythia6.9b.pkl", "rb") as f:
    pythia_6_9b = pickle.load(f)

with open("grid_eval_arc_easy_importance_calibrate_mag_pythia2.8b.pkl", "rb") as f:
    pythia_2_8b = pickle.load(f)

with open("grid_eval_arc_easy_importance_calibrate_mag_pythia1b.pkl", "rb") as f:
    pythia_1b = pickle.load(f)

# Extract accuracies and params from each model
models = {
    "Pythia 6.9B": pythia_6_9b,
    "Pythia 2.8B": pythia_2_8b,
    "Pythia 1B": pythia_1b,
}

# Define markers for each model
markers = {
    "Pythia 6.9B": "o",
    "Pythia 2.8B": "s",  # Square marker
    "Pythia 1B": "D",  # Diamond marker
}

# Get color cycle from plt.rcParams to avoid version-specific errors
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Plotting setup
plt.figure(figsize=(10, 7))

for i, (model_name, model_data) in enumerate(models.items()):
    accs = np.array(model_data["accs"])
    params = np.array(model_data["params"])

    # Get all points
    points = np.array(list(zip(accs, params)))

    # Compute Pareto-optimal points
    pareto_points = pareto_optimal(points)

    # Sort Pareto points by the number of parameters
    sorted_indices = np.argsort(pareto_points[:, 1])  # Sort by params (second column)
    pareto_points = pareto_points[sorted_indices]

    # Scatter all points (without labels) using the same marker for each model
    plt.scatter(
        params,
        accs,
        color=color_cycle[i % len(color_cycle)],
        marker=markers[model_name],
        alpha=0.5,
        s=50,
    )

    # Plot Pareto-optimal points with lines and add labels for legend
    plt.plot(
        pareto_points[:, 1],
        pareto_points[:, 0],
        label=f"{model_name} (Pareto-optimal)",
        color=color_cycle[i % len(color_cycle)],
        marker=markers[model_name],
        linewidth=2,
        markersize=8,
        alpha=0.8,
    )

# Add titles and labels
plt.title("Pareto Fronts for Pythia Models", fontsize=18, fontweight="bold")
plt.xlabel("Number of Parameters", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.grid(True)

# Add legend
plt.legend(title="Models", fontsize=12, title_fontsize=14)

# Apply tight layout
plt.tight_layout()

# Show plot
plt.show()
plt.savefig("pythia_grid.pdf")
