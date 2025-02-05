import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# Adjust overall tick label size
plt.tick_params(labelsize=20)

# Load the data from the pickle file
with open("test_mag.pkl", "rb") as f:
    data = pickle.load(f)

# Extract Magnitude and Params lists
magnitude = data[0]
params = data[1]

# Create bin IDs (1, 2, 3, ..., len(magnitude))
bin_ids = np.arange(1, len(magnitude) + 1)

# Prepare the data for box plots
box_data = []
valid_bin_ids = []

# Iterate over each bin to collect data for the box plot
for bin_id, bin_mag in zip(bin_ids, magnitude):
    if len(bin_mag) >= 50:  # Ensure there are enough observations
        box_data.append(bin_mag[:50])  # Take the first 50 observations
        valid_bin_ids.append(bin_id)  # Collect valid bin IDs

# Create a figure and axis with the specified figsize and dpi
plt.figure(figsize=(12, 8), dpi=150)

# Create the box plot
box = plt.boxplot(
    box_data,
    positions=valid_bin_ids,
    widths=0.6,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="black"),
    whiskerprops=dict(color="black"),
    capprops=dict(color="black"),
    medianprops=dict(color="green"),
)

# Highlight quantiles by plotting them
for i in range(len(box_data)):
    # Calculate quantiles
    quantiles = np.percentile(box_data[i], [25, 50, 75])
    plt.scatter([valid_bin_ids[i]] * len(quantiles), quantiles, color="red", zorder=20)

# Highlight the largest magnitude in each bin
for i, bin_mag in enumerate(box_data):
    max_mag = max(bin_mag)
    plt.scatter(
        valid_bin_ids[i],
        max_mag,
        color="orange",
        s=100,
        edgecolor="black",
        label="Max in bin" if i == 0 else None,
        zorder=10,
    )

# Set x-axis ticks to represent the bin IDs
plt.xticks(valid_bin_ids, fontsize=20)

# Set y-axis ticks to have font size 20
plt.yticks(fontsize=20)

# Use ScalarFormatter for the y-axis to control the scientific notation format
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(20)
# Increase the font size of the scientific notation
ax.yaxis.get_offset_text().set_fontsize(20)

# Add grid lines for readability
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Labels and title
plt.xlabel("Bin ID", fontsize=20)
plt.ylabel("Magnitude", fontsize=20)
plt.title(
    "Quantiles for Architecture Magnitude in Parameter Count Bins",
    fontweight="bold",
    fontsize=22,
)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# Add a legend with contrasting color and position it in the upper left
legend = plt.legend(loc="upper left", fontsize=20, frameon=True)
frame = legend.get_frame()
frame.set_facecolor("lightgrey")  # Change background color
frame.set_edgecolor("black")  # Change edge color

# Adjust layout to prevent clipping
plt.tight_layout()

# Save the figure as a PDF
plt.savefig("boxplots_magnitude.pdf", format="pdf")

# Show the plot
plt.show()
