from modules.utils import get_dataloader
from tqdm import tqdm
import torch
from datasets import load_from_disk

def compute_importance_heads(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
):
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset,  max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prev_end_loc = 0
    model.to(device)
    model.eval()
    head_dict = {}
    # initialize zero score for all batches, all heads
    # initialize zero score for all batches, all heads
    for i in range(model.config.n_head):
        head_dict[str(i)] = [0 for _ in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"attn_{i}"
            q, k, v, mask = model.intermediate_out[k]
            for j in range(model.config.n_head):
                head_act = model.transformer.h[i].attn.scaled_dot_product_attention(
                    q[:, j, :, :].unsqueeze(1),
                    k[:, j, :, :].unsqueeze(1),
                    v[:, j, :, :].unsqueeze(1),
                    mask,
                )  # attention computation for a single head
                head_act = head_act.reshape(head_act.shape[0], max_length, -1)
                head_act = torch.norm(head_act, dim=-1)
                if objective == "norm":
                    # computes score using the norm over B, S
                    head_act_norm = torch.norm(head_act.reshape(-1), dim=-1).item()
                elif objective == "mean":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.absolute(head_act).reshape(-1)
                    ).item()
                elif objective == "norm-mean":
                    # simply mean over B, S
                    head_act_norm = torch.norm(
                        torch.mean(torch.absolute(head_act), dim=-1), dim=-1
                    ).item()
                elif objective == "mean-norm":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.norm(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance-norm":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.norm(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance-mean":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.mean(torch.absolute(head_act), dim=-1), dim=-1
                    ).item()
                elif objective == "mean-variance":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "norm-variance":
                    # simply mean over B, S
                    head_act_norm = torch.norm(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                # aggregate across all layers for each batch and for each head j
                head_dict[str(j)][count] += head_act_norm
        if count + 1 == num_batches:
            break

    for i in range(model.config.n_head):
        head_dict[str(i)] = torch.mean(torch.tensor(head_dict[str(i)]))

    return head_dict


def compute_order_heads(
    max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_importance_scores = compute_importance_heads(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return [
        int(i)
        for i in sorted(
            head_importance_scores, key=head_importance_scores.get, reverse=True
        )
    ]

def compute_order_head_groups(
    max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    head_importance_scores = compute_importance_head_groups(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return [
        int(i)
        for i in sorted(
            head_importance_scores, key=head_importance_scores.get, reverse=True
        )
    ]


def compute_importance_head_groups(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
):
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prev_end_loc = 0
    model.to(device)
    model.eval()
    head_dict = {}
    for i in range(model.config.n_query_groups):
        head_dict[str(i)] = [0 for _ in range(num_batches)]
    n_heads_per_group = model.config.n_head // model.config.n_query_groups
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):  # iterate and sum up scores across layers
            k = f"attn_{i}"
            q, k, v, mask = model.intermediate_out[k]
            for j in range(model.config.n_query_groups):
                head_act = model.transformer.h[i].attn.scaled_dot_product_attention(
                    q[
                        :, j * n_heads_per_group : (j + 1) * n_heads_per_group, :, :
                    ].unsqueeze(1),
                    k[
                        :, j * n_heads_per_group : (j + 1) * n_heads_per_group, :, :
                    ].unsqueeze(1),
                    v[
                        :, j * n_heads_per_group : (j + 1) * n_heads_per_group, :, :
                    ].unsqueeze(1),
                    mask,
                )  # attention computation for a single head
                head_act = head_act.reshape(batch_size, max_length, -1)
                head_act = torch.norm(head_act, dim=-1)
                if objective == "norm":
                    # computes score using the norm over B, S
                    head_act_norm = torch.norm(head_act.reshape(-1), dim=-1).item()
                elif objective == "mean":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.absolute(head_act).reshape(-1)
                    ).item()
                elif objective == "norm-mean":
                    # simply mean over B, S
                    head_act_norm = torch.norm(
                        torch.mean(torch.absolute(head_act), dim=-1), dim=-1
                    ).item()
                elif objective == "mean-norm":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.norm(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance-norm":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.norm(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "variance-mean":
                    # simply mean over B, S
                    head_act_norm = torch.var(
                        torch.mean(torch.absolute(head_act), dim=-1), dim=-1
                    ).item()
                elif objective == "mean-variance":
                    # simply mean over B, S
                    head_act_norm = torch.mean(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                elif objective == "norm-variance":
                    # simply mean over B, S
                    head_act_norm = torch.norm(
                        torch.var(head_act, dim=-1), dim=-1
                    ).item()
                # aggregate across all layers for each batch and for each head j
                head_dict[str(j)][count] += head_act_norm
        if count + 1 == num_batches:
            break

    for i in range(model.config.n_query_groups):
        head_dict[str(i)] = torch.mean(torch.tensor(head_dict[str(i)]))

    return head_dict