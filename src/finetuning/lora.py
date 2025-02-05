# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import dataclasses
import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Literal, Optional, Tuple, Union
import warnings
from syne_tune.config_space import randint, choice
import lightning as L
import torch
from src.utils.sampler import (
    RandomSampler,
    FixGridSampler,
    FixParamGridSampler,
    CalibFixGridSampler,
    ImportanceSampler,
    ImportanceParamGridSampler,
    ImportanceCalibFixGridSampler,
    LlamaGridSampler,
)
from datasets_custom.llamamini import LLaMaMini
from datetime import datetime
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import RunningMean
from whittle.training_strategies.sandwich import SandwichStrategy
from whittle.training_strategies.base_strategy import BaseTrainingStrategy
from litgpt.args import EvalArgs, TrainArgs
from litgpt.data import Alpaca, DataModule
from litgpt.generate.base import generate
from whittle.loss.loss_factory import LossFactory
from litgpt.lora import Config, lora_filter, mark_only_lora_as_trainable
from litgpt.prompts import save_prompt_style
from src.finetuning.sandwich_kd import SandwichStrategy as SandwichStrategyKD
from src.finetuning.standard import StandardStrategy
from src.utils.train_args import FineTuningArgs
from src.utils.search_spaces import search_spaces

# from litgpt.scripts.merge_lora import merge_lora
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    init_out_dir,
    instantiate_torch_optimizer,
    instantiate_bnb_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)
from lora_whittle.lora_gpt import GPT
from lora_whittle.lora_block import LoRABlock as Block
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""This script merges the LoRA weights with the base model"""
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import yaml
from litgpt.lora import Config, lora_filter, merge_lora_weights
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir

from src.utils.utils import plot_validation_metrics, plot_accuracies
from src.finetuning.sandwich import SandwichStrategy


def merge_lora(
    sampling_strategy: str,
    importance_objective: str,
    checkpoint_dir: Path,
    pretrained_checkpoint_dir: Optional[Path] = None,
    precision: Optional[str] = None,
) -> None:
    """Merges the LoRA weights with the base model.

    See ``litgpt finetune lora``.

    Creates a new ``lit_model.pth`` file by merging the LoRA weights (``lit_model.pth.lora``)
    with the original checkpoint weights.

    Arguments:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            ``litgpt finetune lora``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)
    pprint(locals())

    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth.lora")
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / "lit_model.pth").is_file():
        print("LoRA weights have already been merged in this checkpoint.")
        return

    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(
        checkpoint_dir
    )
    precision = precision if precision is not None else lora_precision

    if pretrained_checkpoint_dir is None:
        pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)
    config.fix_head_size = True
    with fabric.init_module(), torch.device("meta"):
        model = GPT(config)
        # we don't care about these to perform merging
        model.cos = None
        model.sin = None

    lora_path = checkpoint_dir / "lit_model.pth.lora"
    if "importance" in sampling_strategy:
        pretrained_checkpoint = torch.load(
        "/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/permuted_model_llama_joint_mean_block_importance.pth"
        )
    else:
        pretrained_checkpoint = torch.load(
            str(pretrained_checkpoint_dir / "lit_model.pth"), mmap=True
        )
    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    # since LoRA finetuning only saves the LoRA weights, we treat the lora weights dtype as the expected dtype
    lora_dtype = next(iter(lora_checkpoint.values())).dtype
    model.to(dtype=lora_dtype, device="cpu")
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {
        k.replace("linear.", ""): v
        for k, v in model.state_dict().items()
        if not lora_filter(k, v)
    }
    save_path = checkpoint_dir / "lit_model.pth"
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(checkpoint_dir / 'lit_model.pth')!r}")


def load_lora_metadata(
    checkpoint_dir: Path,
) -> Tuple[Dict[str, Any], Path, Optional[str]]:
    hparams_file = checkpoint_dir / "hyperparameters.yaml"
    if not hparams_file.is_file():
        raise FileNotFoundError(
            f"The path {str(hparams_file)!r} is not a valid checkpoint directory. It is missing a"
            f" `hyperparameters.yaml` file. Please point to the checkpoint directory that was produced by"
            f" the `litgpt/finetune/lora.py` script."
        )

    with open(hparams_file, "r", encoding="utf-8") as file:
        hparams = yaml.safe_load(file)

    lora_params = {k: v for k, v in hparams.items() if k.startswith("lora_")}
    pretrained_checkpoint_dir = Path(hparams["checkpoint_dir"])
    precision = hparams.get("precision")
    return lora_params, pretrained_checkpoint_dir, precision


def find_resume_path(
    resume: Union[bool, Literal["auto"], Path], out_dir: Path
) -> Optional[Path]:
    resume_path = out_dir / "final" / "lit_model.pth.lora"
    if not resume_path.exists():
        return False
    return resume_path


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/lora"),
    precision: Optional[str] = None,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]
    ] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_query: bool = True,
    lora_key: bool = True,
    lora_value: bool = True,
    lora_projection: bool = True,
    lora_mlp: bool = True,
    lora_head: bool = True,
    lora_emb: bool = True,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=100,
        log_interval=1,
        global_batch_size=64,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=1,
        max_seq_length=None,
    ),
    train_strategy: str = "sandwich-kd",
    search_space_type: str = "hw_gpt_bench",
    sampling_strategy: str = "random",
    eval: EvalArgs = EvalArgs(interval=10, max_new_tokens=100, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "wandb",
    seed: int = 1337,
    access_token: Optional[str] = None,
    n_trials: int = 10000,
    downstream_test_iters: int = 500,
    downstream_dataset: str = "arc_easy",
    importance_objective: str = "norm",
    objective: str = "mag",
    resume: Union[bool, Literal["auto"], Path] = True,
    kd_loss: str = "forward_kl",
    kd_temperature: float = 0.9,
    kd_alpha: float = 0.5,
    kd_beta: float = 2,
    weight_scheme: str = "custom",
    dataset: str = "alpaca",
    weight_supernet_loss: bool = False,
    num_configs: int = 21,
    checkpoint_load_path: str = "/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/permuted_model_llama_joint_mean_block_importance.pth",
    sorted_ids_path: str = "/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/importance_orders_llama_joint_mean_block_importance.pkl",
) -> None:
    """Finetune a model using the LoRA method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        lora_r: The LoRA rank.
        lora_alpha: The LoRA alpha.
        lora_dropout: The LoRA dropout value.
        lora_query: Whether to apply LoRA to the query weights in attention.
        lora_key: Whether to apply LoRA to the key weights in attention.
        lora_value: Whether to apply LoRA to the value weights in attention.
        lora_projection: Whether to apply LoRA to the output projection in the attention block.
        lora_mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
        lora_head: Whether to apply LoRA to output head in GPT.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
    """

    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())
    data_str = dataset
    if data_str == "alpaca":
        data = Alpaca()
    elif data_str == "llamamini":
        data = LLaMaMini()
    else:
        data = None
    print(data)
    devices = parse_devices(devices)
    out_dir = init_out_dir(out_dir)

    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_query=lora_query,
        lora_key=lora_key,
        lora_value=lora_value,
        lora_projection=lora_projection,
        lora_mlp=lora_mlp,
        lora_head=lora_head,
        lora_emb=lora_emb,
    )
    config.fix_head_size = True
    config.tie_embeddings = False
    config.model_type = "gpt"
    now = datetime.now()

    # Additional nanosecond precision by extracting nanoseconds from time.time_ns()
    nanoseconds = time.time_ns() % 1_000_000_000  # Extract only the nanosecond part

    # Create a timestamp with nanosecond precision
    time_string = now.strftime(f"%Y%m%d_%H%M%S")
    search_space = search_spaces[search_space_type](config)

    if train_strategy == "sandwich-kd":
        out_dir = Path(
            f"{config.name}-{train_strategy}-{search_space_type}-{sampling_strategy}-{kd_loss}-{kd_alpha}-{kd_beta}-{kd_temperature}-{weight_scheme}-{weight_supernet_loss}-{data_str}/finetune/lora/"
        )
        id = f"{train_strategy}-{search_space_type}-{sampling_strategy}-{kd_loss}-{kd_alpha}-{kd_beta}-{kd_temperature}-{weight_scheme}-{weight_supernet_loss}-{data_str}-lora"
    else:
        out_dir = Path(
            f"{config.name}-{train_strategy}-{search_space_type}-{sampling_strategy}-{data_str}/finetune/lora/"
        )
        id = f"{train_strategy}-{search_space_type}-{sampling_strategy}-{time_string}-{data_str}-lora"
    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        log_interval=train.log_interval,
        id=id,
        resume=bool(resume),
        config=dict(
            train_strategy=train_strategy,
            search_space_type=search_space_type,
            sampling_strategy=sampling_strategy,
            kd_loss=kd_loss,
            kd_alpha=kd_alpha,
            kd_beta=kd_beta,
            temperature=kd_temperature,
            weight_scheme=weight_scheme,
            data=data_str,
            weight_supernet_loss=weight_supernet_loss,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_emb=lora_emb,
            lora_mlp=lora_mlp,
            lora_head=lora_head,
            lora_projection=lora_projection,
            lr_warmup_steps=train.lr_warmup_steps,
        ),
    )

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32,
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices * num_nodes > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
        '''strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )'''
        strategy = DDPStrategy(find_unused_parameters=True) 
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
        plugins=plugins,
    )
    if sampling_strategy == "random":
        sampler = RandomSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid":
        sampler = FixGridSampler(search_space=search_space, seed=seed)
    elif sampling_strategy == "grid-params":
        sampler = FixParamGridSampler(
            search_space=search_space,
            seed=42,
            n_trials=n_trials,
            num_configs=num_configs,
        )
    elif sampling_strategy == "calibrate":
        sampler = CalibFixGridSampler(
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            seed=seed,
        )
    elif sampling_strategy == "importance-random":
        sampler = ImportanceSampler(
            sorted_ids_path, search_space, seed=seed
        )
    elif sampling_strategy == "importance-grid-params":
        sampler = ImportanceParamGridSampler(
            sorted_ids_path=sorted_ids_path,
            search_space=search_space,
            seed=42,
            num_configs=num_configs,
            n_trials=n_trials,
        )
    elif sampling_strategy == "importance-calibrate":
        sampler = ImportanceCalibFixGridSampler(
            objective=objective,
            importance_objective=importance_objective,
            sorted_ids_path=sorted_ids_path,
            checkpoint_dir=checkpoint_dir,
            search_space_type=search_space_type,
            search_space=search_space,
            num_configs=num_configs,
            seed=seed,
        )
    elif sampling_strategy == "llama-grid":
        sampler = LlamaGridSampler(
            sorted_ids_path=os.path.join(checkpoint_dir, "sorted_ids.pkl"), seed=seed
        )
    loss_factory = LossFactory(
        alpha=kd_alpha,
        beta=kd_beta,
        temperature=kd_temperature,
        weight_scheme=weight_scheme,
    )
    if train_strategy == "sandwich-kd":
        strategy = SandwichStrategyKD(
            loss_function=loss_factory,
            lora=True,
            sampler=sampler,
            weight_supernet_loss=weight_supernet_loss,
        )
    elif train_strategy == "sandwich":
        strategy = SandwichStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
            weight_supernet_loss=weight_supernet_loss,
        )

    elif train_strategy == "standard":
        strategy = StandardStrategy(
            loss_function=chunked_cross_entropy,
            lora=True,
            sampler=sampler,
        )

    strategy.fabric = fabric
    strategy.gradient_accumulation_step = train.gradient_accumulation_iters(devices)
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        devices,
        seed,
        config,
        data,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        optimizer,
        sampling_strategy,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        importance_objective,
        resume,
        kd_loss,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: Union[str, Dict],
    sampling_strategy: str,
    strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    importance_objective: str,
    resume: bool,
    kd_loss: str,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, data, tokenizer, train)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(
        devices
    )
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    if "importance" in sampling_strategy:
        checkpoint_path = "/hkfs/work/workspace/scratch/fr_rs1131-peftprune/compressing_llms/checkpoints/meta-llama/Meta-Llama-3.1-8B/permuted_model_llama_joint_mean_block_importance.pth"
    else:
        checkpoint_path = checkpoint_dir / f"lit_model.pth"
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
        if "grid-params" in sampling_strategy:
            print("Initializing params grid....")
            strategy.sampler.initialize_grid(model)
            print("Requested configs:", len(strategy.sampler.values))
            print("Grid Size", len(strategy.sampler.grid))
    model.name_or_path = checkpoint_dir
    mark_only_lora_as_trainable(model)

    fabric.print(
        f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}"
    )
    fabric.print(
        f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}"
    )

    model = fabric.setup_module(model)

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer = instantiate_bnb_optimizer(optimizer, model.parameters())
    else:
        optimizer = instantiate_torch_optimizer(optimizer, model.parameters())

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps
    )
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "step_count": 0,
    }
    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    if resume:
        resume = find_resume_path(resume, out_dir)
        if resume:
            fabric.load(resume, state)
    train_time = time.perf_counter()
    fit(
        fabric,
        state,
        model,
        optimizer,
        scheduler,
        train_dataloader,
        val_dataloader,
        devices,
        checkpoint_dir,
        out_dir,
        train,
        eval,
        data,
        strategy,
        downstream_dataset,
        downstream_test_iters,
        resume,
        kd_loss,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    if eval.final_validation:
        val_loss_largest, val_loss_medium, val_loss_smallest = validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            True,
            strategy.sampler,
        )
        metrics = {"val_loss": val_loss_largest, "val_ppl": math.exp(val_loss_largest)}
        fabric.log_dict(metrics)
        fabric.print(
            f"Final evaluation | val loss: {val_loss_largest.item():.3f} | val ppl: {math.exp(val_loss_largest):.3f}"
        )

    # Save the final LoRA checkpoint at the end of training
    save_path = out_dir / "final" / "lit_model.pth.lora"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_lora_checkpoint(fabric, model, save_path)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_path.parent)
        save_hyperparameters(setup, save_path.parent)
        save_prompt_style(data.prompt_style, save_path.parent)
        merge_lora(
            checkpoint_dir=save_path.parent,
            importance_objective=importance_objective,
            sampling_strategy=sampling_strategy,
        )


def fit(
    fabric: L.Fabric,
    state: Dict,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    train_strategy: BaseTrainingStrategy,
    downstream_dataset: str,
    downstream_test_iters: int,
    resume: bool,
    kd_loss: str,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)
    longest_seq_length, longest_seq_ix = get_longest_seq_length(
        ConcatDataset([train_dataloader.dataset, val_dataloader.dataset])
    )
    model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    fabric.print(
        f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
        f" {model.max_seq_length} and context length is {model.config.block_size}"
    )

    if eval.initial_validation:
        val_loss_largest, val_loss_medium, val_loss_smallest = validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=len(val_dataloader)),
            verbose=True,
            sampler=train_strategy.sampler,
        )
        val_loss_largest = f"{val_loss_largest:.3f}"
        val_loss_medium = f"{val_loss_medium:.3f}"
        val_loss_smallest = f"{val_loss_smallest:.3f}"
    else:
        fabric.print("Verifying settings ...")
        validate(
            fabric,
            model,
            val_dataloader,
            dataclasses.replace(eval, max_iters=2),
            verbose=False,
            sampler=train_strategy.sampler,
        )  # sanity check
        val_loss_largest = "n/a"
        val_loss_medium = "n/a"
        val_loss_smallest = "n/a"
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                fabric.print(f"Resuming dataset: {resume_iter} / {initial_iter}")
        fabric.barrier()
        fabric.print(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f} seconds to reach iteration"
            f" {initial_iter}."
        )
    # throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices), sync_on_compute=False
    ).to(fabric.device)
    max_steps = train.max_steps or float("inf")
    step_count = 0
    iter_num = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    while state["step_count"] < max_steps and train_iterator.epoch < train.epochs:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = next(train_iterator)
        input_ids, targets = batch["input_ids"], batch["labels"]

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices) != 0
        )
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = train_strategy(
                model,
                input_ids,
                targets,
                train.gradient_accumulation_iters(devices),
                kd_loss,
            )

        running_loss.update(loss)

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            state["step_count"] += 1

        total_lengths += input_ids.numel()
        if state["iter_num"] % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            # throughput.update(
            #    time=t1 - total_t0,
            #    batches=state["iter_num"],
            #    samples=state["iter_num"] * train.micro_batch_size,
            #    lengths=total_lengths,
            # )
            # throughput.compute_and_log(step=state["iter_num"])
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": state["iter_num"]
                * train.micro_batch_size
                * model.config.block_size,
                "total_tokens": (
                    state["iter_num"]
                    * train.micro_batch_size
                    * model.config.block_size
                    * fabric.world_size
                ),
                "learning_rate": scheduler.get_last_lr()[0],
            }
            if isinstance(val_loss_largest, torch.Tensor):
                val_loss_largest = f"{val_loss_largest:.3f}"
                val_loss_medium = f"{val_loss_medium:.3f}"
                val_loss_smallest = f"{val_loss_smallest:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch'] + 1} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val_loss_largest: {val_loss_largest} |"
                f" val_loss_medium: {val_loss_medium} |"
                f" val_loss_smallest: {val_loss_smallest} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            t0 = time.perf_counter()
            val_loss_largest, val_loss_medium, val_loss_smallest = validate(
                fabric, model, val_dataloader, eval, True, train_strategy.sampler
            )
            generate_example(fabric, model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(
                f"iter {state['iter_num']}: val loss largest{val_loss_largest.item():.4f},val loss medium{val_loss_medium.item():.4f}, val loss  smallest{val_loss_smallest.item():.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {
                "val_loss_largest": val_loss_largest,
                "val_ppl_largest": math.exp(val_loss_largest),
                "val_loss_medium": val_loss_medium,
                "val_ppl_medium": math.exp(val_loss_medium),
                "val_loss_smallest": val_loss_smallest,
                "val_ppl_smallest": math.exp(val_loss_smallest),
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()
        if not is_accumulating and state["step_count"] % downstream_test_iters == 0:
            t0 = time.perf_counter()
            acc_largest, acc_medium, acc_smallest = test_downstream(
                fabric, model, downstream_dataset, train_strategy.sampler, out_dir
            )
            t1 = time.perf_counter() - t0
            fabric.print(
                f"iter {state['iter_num']}: acc largest{acc_largest:.4f},acc medium{acc_medium:.4f}, acc smallest{acc_smallest:.4f}, val time: {t1 * 1000:.2f} ms"
            )
            metrics = {
                "acc_largest": acc_largest,
                "acc_medium": acc_medium,
                "acc_smallest": acc_smallest,
            }
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            checkpoint_file = out_dir / f"final" / "lit_model.pth.lora"
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                save_prompt_style(data.prompt_style, checkpoint_file.parent)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: GPT,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    verbose: bool = True,
    sampler=None,
) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    val_loss_largest, val_loss_middle, val_loss_smallest = plot_validation_metrics(
        model, val_dataloader, eval, sampler
    )
    model.train()
    return val_loss_largest, val_loss_middle, val_loss_smallest


@torch.no_grad()
def test_downstream(
    fabric: L.Fabric,
    model: GPT,
    dataset: str = "arc_easy",
    sampler=None,
    checkpoint_dir=None,
) -> torch.Tensor:
    model.eval()
    acc_largest, acc_middle, acc_smallest = plot_accuracies(
        model, sampler, dataset, checkpoint_dir
    )
    model.train()
    return acc_largest, acc_middle, acc_smallest


@torch.no_grad()
def generate_example(
    fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule
):
    instruction = (
        "Recommend a movie for me to watch during the weekend and explain the reason."
    )
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()
    model.reset_super_network()
    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model,
            encoded,
            max_returned_tokens=max_returned_tokens,
            temperature=0.8,
            eos_id=tokenizer.eos_id,
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(output)
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [scheduler1, scheduler2], milestones=[warmup_steps]
    )


def get_dataloaders(
    fabric: L.Fabric, data: DataModule, tokenizer: Tokenizer, train: TrainArgs
) -> Tuple[DataLoader, DataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        max_seq_length=train.max_seq_length,
    )
    with fabric.rank_zero_first():
        data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )
    return train_dataloader, val_dataloader


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def save_lora_checkpoint(
    fabric: L.Fabric, model: torch.nn.Module, file_path: Path
) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [
        (train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])
    ]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if not train.epochs and not train.max_steps:
        issues.append(
            f"{__file__} requires either epochs or max_steps to be set. This is set in {train}"
        )
    if issues:
        raise ValueError("\n".join(issues))


if __name__ == "__main__":
    from jsonargparse import CLI

    # setup("EleutherAI/pythia-1b", search_space_gpt)
    CLI(setup)
