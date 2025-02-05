import torch
from tqdm import tqdm
from whittle.metrics.parameters import compute_parameters
from datasets import load_dataset
import pickle
from whittle.metrics.mag import compute_weight_magnitude
from syne_tune.config_space import Categorical, Domain
from modules.utils import evaluate_wikitext

def constrained_search(
    sampler, search_space, params_min, params_max, model, trials=1000
):
    for _ in range(trials):
        config = sampler.sample()
        model.set_sub_network(**config)
        params = compute_parameters(model)
        model.reset_super_network()
        if params >= params_min and params < params_max:
            return [config, params]


def get_smallest_params(search_space, model):
    config = {}
    for hp_name, hp in search_space.config_space.items():
        if isinstance(hp, Categorical):
            config[hp_name] = min(hp.categories)
        else:
            config[hp_name] = hp.lower

    model.set_sub_network(**search_space.cast(config))
    params = compute_parameters(model)
    model.reset_super_network()
    return params


def compute_perplexity_wikitext(
    model,
    search_space,
    sampler,
    tokenizer,
    search_space_name,
    num_archs=50,
    num_bins=22,
    max_length = 1024,
    batch_size = 4,
    num_batches = 100
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = model.name_or_path + "/" + search_space_name + "_calibration_stats_ppl.pkl"
    values = [(i) / num_bins for i in range(num_bins + 1)]
    ppls = []
    configs = []
    params_all = []
    model.reset_super_network()
    u = compute_parameters(model)
    l = get_smallest_params(search_space, model)
    params_min = l
    for value in values[1:]:
        params_max = int(value * (u - l) + l)
        params_subset = []
        ppls_subset = []
        configs_subset = []
        for _ in range(num_archs):
            nlls = []
            prev_end_loc = 0

            search_arch = constrained_search(
                sampler, search_space, params_min, params_max, model
            )
            if search_arch is not None:
                (config_sampled, params) = search_arch
            else:
                continue
            model.set_sub_network(**config_sampled)
            ppl = evaluate_wikitext(max_length,model,tokenizer,batch_size,num_batches)
            model.reset_super_network()
            ppls_subset.append(ppl)
            params_subset.append(params)
            configs_subset.append(config_sampled)
        params_min = params_max
        if params_subset != []:
            params_all.append(params_subset)
            ppls.append(ppls_subset)
            configs.append(configs_subset)
    with open(save_path, "wb") as f:
        pickle.dump([ppls, params_all, configs], f)
    return ppls, params_all, configs


def compute_magnitude(
    model,
    search_space,
    sampler,
    tokenizer,
    search_space_name,
    num_archs=50,
    num_bins=22,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = (
        model.name_or_path + "/" + search_space_name + "_calibration_stats_mag.pkl"
    )
    max_length = 512
    values = [(i) / num_bins for i in range(num_bins + 1)]
    magnitudes = []
    configs = []
    params_all = []
    model.reset_super_network()
    u = compute_parameters(model)
    l = get_smallest_params(search_space, model)
    params_min = l
    for value in values[1:]:
        params_max = int(value * (u - l) + l)
        params_subset = []
        magnitude_subset = []
        configs_subset = []
        for _ in range(num_archs):
            nlls = []
            prev_end_loc = 0

            search_arch = constrained_search(
                sampler, search_space, params_min, params_max, model
            )
            if search_arch is not None:
                (config_sampled, params) = search_arch
            else:
                continue
            model.set_sub_network(**config_sampled)
            magnitude = compute_weight_magnitude(model)
            model.reset_super_network()
            magnitude_subset.append(magnitude)
            params_subset.append(params)
            configs_subset.append(config_sampled)
        params_min = params_max
        if params_subset != []:
            params_all.append(params_subset)
            magnitudes.append(magnitude_subset)
            configs.append(configs_subset)
            print(magnitudes)
    with open(save_path, "wb") as f:
        pickle.dump([magnitudes, params_all, configs], f)
    return magnitudes, params_all, configs


def get_architecture_grid_mag(sampler, magnitudes, params, arch_configs, num_bins=21):
    grid = []
    with open("test_mag.pkl", "wb") as f:
        pickle.dump([magnitudes, params, arch_configs], f)
    if sampler.get_smallest_sub_network() not in grid:
        grid.append(sampler.get_smallest_sub_network())
    for i in range(len(magnitudes)):
        magnitude_subset = magnitudes[i]
        params_subset = params[i]
        archs_subset = arch_configs[i]
        sorted_mag, sorted_params, sorted_archs = zip(
            *sorted(
                zip(magnitude_subset, params_subset, archs_subset), key=lambda x: -x[0]
            )
        )
        if sorted_archs[0] not in grid:
            grid.append(sorted_archs[0])  # add arch with largest mag in parameter
    if sampler.get_largest_sub_network() not in grid:
        grid.append(sampler.get_largest_sub_network())
    return grid


def get_architecture_grid(sampler, ppls, params, arch_configs, num_bins=21):
    grid = []
    if sampler.get_smallest_sub_network() not in grid:
        grid.append(sampler.get_smallest_sub_network())
    for i in range(len(ppls)):
        ppl_subset = ppls[i]
        params_subset = params[i]
        archs_subset = arch_configs[i]
        sorted_ppls, sorted_params, sorted_archs = zip(
            *sorted(zip(ppl_subset, params_subset, archs_subset), key=lambda x: x[0])
        )
        if sorted_archs[0] not in grid:
            grid.append(sorted_archs[0])  # add arch with smallest ppl in parameter
    if sampler.get_largest_sub_network() not in grid:
        grid.append(sampler.get_largest_sub_network())
    return grid
