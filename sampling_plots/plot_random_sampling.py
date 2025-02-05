import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from llm_compression.sampler import RandomSampler, FixParamGridSampler
from llm_compression.search_spaces import search_spaces
from whittle.models.gpt import GPT
from whittle.metrics.parameters import compute_parameters_sub_network_gpt
from litgpt import Config

# Adjust tick label size
plt.tick_params(labelsize=20)

# Load the configuration
config = Config.from_file(
    "/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/checkpoints/meta-llama/Meta-Llama-3.1-8B/model_config.yaml"
)
config.fix_head_size = True

# Define the search space and model
search_space = search_spaces["llama2"](config)
config.model_type = "gpt"
model = GPT(config)

# Random Sampling
sampler = RandomSampler(search_space)
params = []

# Sample configurations and compute parameters (random sampling)
for i in range(1000):
    sample_config = sampler.sample()
    model.set_sub_network(**sample_config)
    params.append(compute_parameters_sub_network_gpt(model))

# Grid Sampling
sampler = FixParamGridSampler(search_space, seed=42)
params_grid = []
sampler.initialize_grid(model)

# Sample configurations and compute parameters (grid sampling)
for i in range(1000):
    sample_config = sampler.sample()
    model.set_sub_network(**sample_config)
    params_grid.append(compute_parameters_sub_network_gpt(model))

# Plot for Random Sampling
plt.figure(figsize=(10, 6), dpi=150)
plt.hist(params, bins=14, alpha=0.7, color="blue", edgecolor="black")

# Add custom scientific notation formatting for y-axis
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(20)

# Titles, labels, and grid
plt.title("Distribution of Parameters: Random Sampling", fontsize=22, weight="bold")
plt.xlabel("Number of Parameters", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle="--", alpha=0.6)

# Save and show plot
plt.savefig("random_sampling_parameters.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot for Grid Sampling
plt.figure(figsize=(10, 6), dpi=150)
plt.hist(params_grid, bins=14, alpha=0.7, color="orange", edgecolor="black")

# Add custom scientific notation formatting for y-axis
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(20)

# Titles, labels, and grid
plt.title("Distribution of Parameters: Grid Sampling", fontsize=22, weight="bold")
plt.xlabel("Number of Parameters", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle="--", alpha=0.6)

# Save and show plot
plt.savefig("grid_sampling_parameters.pdf", format="pdf", bbox_inches="tight")
plt.show()
