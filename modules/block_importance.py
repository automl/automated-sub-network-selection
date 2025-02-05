from modules.utils import get_dataloader
from tqdm import tqdm
import torch
from datasets import load_from_disk

def compute_block_importance(
    max_length, objective, model, tokenizer, batch_size=32, num_batches=10
):
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset,  max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    bi_dict = {}
    model.reset_super_network()
    # initialize zero score for all batches, all layers
    for i in range(model.config.n_layer):
        bi_dict[str(i)] = [0 for _ in range(num_batches)]
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            _ = model(input_ids)
        for i in range(model.config.n_layer):
            mat_1 = model.intermediate_out[f"block_{i}"].reshape(
                -1, model.config.n_embd
            )  # layer input, shape B*S, Emb
            mat_2 = model.intermediate_out[f"block_{i+1}"].reshape(
                -1, model.config.n_embd
            )  # layer output, shape B*S, Emb

            for j in range(mat_1.shape[0]):  # index over features of each word i.e. B*S
                xtx = torch.sum(mat_1[j, :] * mat_2[j, :])
                norm1 = torch.norm(mat_1[j, :])
                norm2 = torch.norm(mat_2[j, :])
                bi_dict[str(i)][count] += xtx / (
                    norm1 * norm2
                )  # sum over word feature similarity
            bi_dict[str(i)][count] = (
                bi_dict[str(i)][count].item() / mat_1.shape[0]
            )  # divide by total number of words
        if count + 1 == num_batches:
            break

    # Calculate the inverted block importance aggregating over batches
    bi_dict = {
        str(i): 1 - torch.mean(torch.tensor(bi_dict[str(i)]))
        for i in range(model.config.n_layer)
    }
    return bi_dict


def compute_order_block_importance(
    max_seq_len, objective, model, tokenizer, batch_size=32, num_batches=10
):
    model.reset_super_network()
    scores = compute_block_importance(
        max_seq_len, objective, model, tokenizer, batch_size, num_batches
    )
    return [
        int(i) for i in sorted(scores, key=scores.get, reverse=True)
    ]  # higher the score more important the block
