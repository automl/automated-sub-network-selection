import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders
import math
from tqdm import tqdm

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    "IFV": lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    "WIFV": lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp
    * torch.sum(subset[name].weight.data.pow(2), dim=0),
    "WIFN": lambda wrapped_layers, subset, name: (
        torch.abs(subset[name].weight.data)
        * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1, -1)))
    ).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in getattr(model, "hf_device_map", {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (2048, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {"i": 0, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    position_ids = cache["position_ids"]
    model.config.use_cache = use_cache

    return inps, outs, position_ids


def compress(
    layer,
    attn_mask,
    mlp_mask,
    attn_mean_inp,
    mlp_mean_inp,
    device,
    bias=True,
    args=None,
):
    """
    Compress a model layer by masking or pruning based on the given masks.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.

    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    # Real Pruning
    # Attention Weight Pruning
    if attn_mask is not None:
        retain_heads_kv = torch.count_nonzero(attn_mask)
        retain_heads_qo = torch.count_nonzero(attn_mask).repeat_interleave(
            args.gqa_groups
        )
        attn_mask_kv = attn_mask.repeat_interleave(args.head_dim)
        attn_mask_qo = attn_mask.repeat_interleave(args.gqa_groups).repeat_interleave(
            args.head_dim
        )

        # Prune the query, key and value projection weights
        # We reduce the size of the weights based on the attention mask
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[
            torch.where(attn_mask_qo)[0]
        ]
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[
            torch.where(attn_mask_kv)[0]
        ]
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[
            torch.where(attn_mask_kv)[0]
        ]

        # Update output dimensions of q, k, v projections based on remaining heads
        layer.self_attn.q_proj.out_features = attn_mask_qo.sum().item()
        layer.self_attn.k_proj.out_features = attn_mask_kv.sum().item()
        layer.self_attn.v_proj.out_features = attn_mask_kv.sum().item()

        output_weight = layer.self_attn.o_proj.weight.data

        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = (
                attn_mean_inp.to(device) * ~attn_mask_qo.to(device)
            ) @ output_weight.T

        # Prune the output projection weight
        output_weight = layer.self_attn.o_proj.weight.data[
            :, torch.where(attn_mask_qo)[0]
        ]
        # Update layer configurations for the new output shape after pruning
        layer.self_attn.num_heads = retain_heads_qo
        layer.self_attn.hidden_size = retain_heads_qo * args.head_dim
        if args.gqa_groups > 1:
            layer.self_attn.num_key_value_heads = retain_heads_kv

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            layer.self_attn.o_proj.in_features = attn_mask_qo.sum().item()
            # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            layer.self_attn.o_proj.bias.data = output_bias

        # Assign the pruned weights
        layer.self_attn.o_proj.weight.data = output_weight

    # MLP Weight Pruning
    if mlp_mask is not None:
        # Prune the up and gate projection weights
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[
            torch.where(mlp_mask)[0]
        ]
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[
            torch.where(mlp_mask)[0]
        ]

        # Update output dimensions of up and gate projections based on the mlp mask
        layer.mlp.up_proj.out_features = mlp_mask.sum().item()
        layer.mlp.gate_proj.out_features = mlp_mask.sum().item()

        output_weight = layer.mlp.down_proj.weight.data
        layer.mlp.intermediate_size = mlp_mask.sum().item()
        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = (
                mlp_mean_inp.to(device) * ~mlp_mask.to(device)
            ) @ output_weight.T

        # Prune the down projection weight
        output_weight = layer.mlp.down_proj.weight.data[:, torch.where(mlp_mask)[0]]

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            layer.mlp.down_proj.in_features = mlp_mask.sum().item()
            # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            layer.mlp.down_proj.bias.data = output_bias

        # Assign the pruned weights
        layer.mlp.down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Our FLAP Pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, position_ids = prepare_calibration_input(model, dataloader, device)
    layers = model.model.layers

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    # Split into sub-problems, separate statistics for each module
    for i in tqdm(
        range(args.start_pruning_layer_idx, len(layers)), desc="Processing layers"
    ):
        layer = layers[i]
        subset = {}
        subset.update({"self_attn.o_proj": find_layers(layer)["self_attn.o_proj"]})
        subset.update({"mlp.down_proj": find_layers(layer)["mlp.down_proj"]})

        if (
            f"model.layers.{i}" in getattr(model, "hf_device_map", {})
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, position_ids = inps.to(dev), outs.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if name == "self_attn.o_proj":
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(
                    wrapped_layers[name].baseline_inp.type(torch.half)
                )
            wrapped_layers[name].free()

        inps, outs = outs, inps  # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(
        x, axis=1, keepdim=True
    )

    if args.structure in ["AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        attn_metric = attn_metric.reshape(
            len(layers) - args.start_pruning_layer_idx,
            -1,
            args.head_dim * args.gqa_groups,
        ).mean(dim=2)

        mlp_metric = torch.stack(mlp_metric_list)
        mlp_metric = standarlization(mlp_metric)

        prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
        sorted_prune, indices = torch.sort(prune_metric, descending=True)
        compression_weight = torch.ones_like(indices)
        compression_weight[indices < attn_metric.numel()] = 512.0 / 3
        threshold = sorted_prune[
            torch.argmin(
                torch.abs(
                    torch.cumsum(compression_weight, 0)
                    - torch.sum(compression_weight) * (1 - args.pruning_ratio)
                )
            )
        ]
        attn_mask = attn_metric > threshold
        mlp_mask = mlp_metric > threshold

    for idx in range(len(layers) - args.start_pruning_layer_idx):
        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            compress(
                model.model.layers[idx],
                attn_mask[idx],
                None,
                attn_baseline_inp_list[idx],
                None,
                model.hf_device_map[f"model.layers.{idx}"],
                args=args,
            )
        else:
            compress(
                model.model.layers[idx],
                attn_mask[idx],
                None,
                attn_baseline_inp_list[idx],
                None,
                device,
                args=args,
            )

        if f"model.layers.{i}" in getattr(model, "hf_device_map", {}):
            compress(
                model.model.layers[idx],
                None,
                mlp_mask[idx],
                None,
                mlp_baseline_inp_list[idx],
                model.hf_device_map[f"model.layers.{idx}"],
            )
        else:
            compress(
                model.model.layers[idx],
                None,
                mlp_mask[idx],
                None,
                mlp_baseline_inp_list[idx],
                device,
            )

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.

    Args:
        model (nn.Module): The model to check.

    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if "self_attn" in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params
