import matplotlib.pyplot as plt
from llm_compression.sampler import FixParamGridSampler
from llm_compression.search_spaces import search_spaces
from whittle.models.gpt import GPT
from whittle.metrics.parameters import (
    compute_parameters,
    compute_parameters_sub_network_gpt,
)
from litgpt import Config

# Load the configuration
config = Config.from_file(
    str(
        f"/p/project/projectnucleus/sukthanker1/importance/llm_compression_experiments/checkpoints/meta-llama/Meta-Llama-3.1-8B/model_config.yaml"
    )
)
config.fix_head_size = True

# Define the search space and model
search_space = search_spaces["llama2"](config)
config.model_type = "gpt"
model = GPT(config)

# Initialize the sampler
sampler = FixParamGridSampler(search_space, seed=42)
params = []
sampler.initialize_grid(model)
print(len(sampler.grid))
# Sample configurations and compute parameters
for i in range(1000):
    sample_config = sampler.sample()
    model.set_sub_network(**sample_config)
    params.append(compute_parameters_sub_network_gpt(model))

# Plot the histogram of the sampled parameters
# Plot the histogram of the sampled parameters
# Set up the figure for higher quality plotting
plt.figure(figsize=(8, 6), dpi=150)

# Plot the histogram of the sampled parameters
plt.hist(params, bins=20, alpha=0.75, color="blue", edgecolor="black")

# Improve titles and labels for better readability
plt.title("Architectures on a Uniform Parameter Grid", fontsize=22, weight="bold")
plt.xlabel("Number of Parameters", fontsize=20)
plt.ylabel("Frequency", fontsize=20)

# Add grid for better visual guidance
plt.grid(True, linestyle="--", alpha=0.6)

# Save the plot with higher resolution
plt.savefig("grid_sampling.pdf", format="pdf", bbox_inches="tight")
