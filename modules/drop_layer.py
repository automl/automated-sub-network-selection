from modules.utils import get_dataloader
from tqdm import tqdm
import torch
from datasets import load_from_disk

def compute_ppl_block(max_length, model, tokenizer, batch_size=32, num_batches=10):
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset,  max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)  # output with a dropped layer
            neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        nlls.append(neg_log_likelihood)
        if count + 1 == num_batches:
            break

    return torch.exp(torch.stack(nlls).mean()).item()  # compute perplextity


def compute_order_layers_ppl(
    max_seq_len, model, tokenizer, largest_model_config, batch_size=32, num_batches=10
):
    model.reset_super_network()
    layer_importance_scores = {}
    sub_network_n_layers = largest_model_config["sub_network_n_layers"]-1
    del largest_model_config["sub_network_n_layers"]
    for layer in range(model.config.n_layer):
        model.set_sub_network(
            **largest_model_config, sampled_layer_indices = [i for i in range(model.config.n_layer) if i != layer], sub_network_n_layers = sub_network_n_layers
        )  # drop layer corresponding to index_block
        score = compute_ppl_block(
            max_seq_len, model, tokenizer, batch_size, num_batches
        )
        layer_importance_scores[str(layer)] = score
    return [
        int(i)
        for i in sorted(
            layer_importance_scores, key=layer_importance_scores.get, reverse=True
        )
    ]  # reverse list as higher the perplexity more important the block
