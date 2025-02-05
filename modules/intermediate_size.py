from modules.utils import get_dataloader
from tqdm import tqdm
import torch
from datasets import load_from_disk

def compute_importance_intermediate_size(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
):
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset,  max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prev_end_loc = 0
    model.to(device)
    model.eval()
    mlp_scores = {}
    # initialize zero score for all batches, all neurons
    # initialize zero score for all batches, all neurons
    for i in range(model.config.intermediate_size):
        mlp_scores[f"{i}"] = [0 for i in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():  # don't compute grad, save memory
            _ = model(input_ids)  # save activations in forward
            # intermediate_out is a dictionary saving intermediate activations
            for k in model.intermediate_out:
                if "mlp" in k:  # filter out only MLP keys (across layers)
                    for i in range(model.config.intermediate_size):
                        matrix_x_fc = model.intermediate_out[k].reshape(
                            -1, model.config.intermediate_size
                        )[:, i]  # extract the output corresponding to ith neuron
                        if objective == "norm":
                            # computes score using the norm over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.norm(matrix_x_fc.reshape(-1)).item()
                            )
                        elif objective == "mean":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.mean(
                                    torch.absolute((matrix_x_fc)).reshape(-1)
                                ).item()
                            )
                        elif objective == "norm-mean":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.norm(
                                    torch.mean(torch.absolute(matrix_x_fc), dim=-1),
                                    dim=-1,
                                ).item()
                            )
                        elif objective == "mean-norm":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.mean(
                                    torch.norm(matrix_x_fc, dim=-1), dim=-1
                                ).item()
                            )
                        elif objective == "variance":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.var(
                                    torch.var(matrix_x_fc, dim=-1), dim=-1
                                ).item()
                            )
                        elif objective == "variance-norm":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.var(
                                    torch.norm(matrix_x_fc, dim=-1), dim=-1
                                ).item()
                            )
                        elif objective == "variance-mean":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.var(
                                    torch.mean(torch.absolute(matrix_x_fc), dim=-1),
                                    dim=-1,
                                ).item()
                            )
                        elif objective == "mean-variance":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.mean(
                                    torch.var(matrix_x_fc, dim=-1), dim=-1
                                ).item()
                            )
                        elif objective == "norm-variance":
                            # simply mean over B, S
                            mlp_scores[f"{i}"][count] = (
                                mlp_scores[f"{i}"][count]
                                + torch.norm(
                                    torch.var(matrix_x_fc, dim=-1), dim=-1
                                ).item()
                            )
        if count + 1 == num_batches:
            break
    # compute mean over batches
    for i in range(model.config.intermediate_size):
        mlp_scores[str(i)] = torch.mean(torch.tensor(mlp_scores[str(i)]))
    return mlp_scores


def compute_order_intermediate_dims(
    max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    mlp_importance_scores = compute_importance_intermediate_size(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return [
        int(i)
        for i in sorted(
            mlp_importance_scores, key=mlp_importance_scores.get, reverse=True
        )
    ]
