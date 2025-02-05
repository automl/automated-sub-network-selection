from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import pickle

from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import pickle


# Tokenize each example with dynamic sequence length
def tokenize_function(examples, tokenizer, seq_len):
    encodings = tokenizer("\n\n".join(examples["text"]), return_tensors="pt")
    seq_len_orig = encodings.input_ids.size(1)
    input_ids_li = []
    target_ids_li = []

    for begin_loc in range(0, seq_len_orig, seq_len):
        end_loc = min(begin_loc + seq_len, seq_len_orig)
        if end_loc != begin_loc + seq_len:  # ignore last batch
            break
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = torch.zeros_like(input_ids)
        target_ids[:, 0:-1] = input_ids[:, 1:]
        target_ids[:, -1] = -100
        # target_ids[:, -1] = tokenizer.pad_token_id  # Target padding
        input_ids_li.append(torch.squeeze(input_ids))
        target_ids_li.append(torch.squeeze(target_ids))

    return {
        "input_ids": input_ids_li,
        "labels": target_ids_li,
    }


# Dataset class
class TextDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.squeeze(torch.tensor(self.input_ids[idx])).long(),
            "labels": torch.squeeze(torch.tensor(self.labels[idx])).long(),
        }


# Dataloader function
def get_dataloader(tokenizer, seq_len=512, batch_size=32):
    # Load the wikitext test dataset
    test = load_dataset("zwhe99/commonsense_170k")[
        "train"
    ]  # "wikitext-2-raw-v1", split="test")
    # print(test.keys())
    test = test.map(lambda x: {"text": f"{x['instruction']} {x['output']}"})

    # Ensure pad_token is set for LlamaTokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else tokenizer.eos_token_id
        )

    # Apply tokenization
    tokenized_dataset = tokenize_function(test, tokenizer, seq_len)
    # Flatten the tokenized dataset
    # input_ids = [item for sublist in tokenized_dataset["input_ids"] for item in sublist]
    # labels = [item for sublist in tokenized_dataset["labels"] for item in sublist]

    # Initialize Dataset and DataLoader
    dataset = TextDataset(tokenized_dataset["input_ids"], tokenized_dataset["labels"])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def evaluate_wikitext(max_length, model, tokenizer, batch_size, num_batches):
    dataloader = get_dataloader(tokenizer, max_length, batch_size)
    nlls = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    torch.manual_seed(2809)
    for count, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Each batch of size B, S
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)["logits"]
            neg_log_likelihood = torch.nn.CrossEntropyLoss()(
                outputs.view(-1, outputs.size(-1)), target_ids.view(-1)
            )
        nlls.append(neg_log_likelihood)
        if count + 1 == num_batches:
            break
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def permute_norm(embedding_order, norm):
    if norm is None:
        return
    if isinstance(norm, LayerNorm):
        norm.weight.data = norm.weight.data[embedding_order]
        norm.bias.data = norm.bias.data[embedding_order]
    elif isinstance(norm, RMSNorm):
        norm.weight.data = norm.weight.data[embedding_order]
    else:
        raise ValueError("Norm not supported")


def permute_mlp(embedding_order, mlp_order, mlp):
    if isinstance(mlp, GptNeoxMLP):
        mlp.fc.weight.data = mlp.fc.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc.bias is not None:
            mlp.fc.bias.data = mlp.fc.bias.data[mlp_order]
    elif isinstance(mlp, (LLaMAMLP, GemmaMLP)):
        mlp.fc_1.weight.data = mlp.fc_1.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc_1.bias is not None:
            mlp.fc_1.bias.data = mlp.fc_1.bias.data[mlp_order]
        mlp.fc_2.weight.data = mlp.fc_2.weight.data[mlp_order, :][:, embedding_order]
        if mlp.fc_2.bias is not None:
            mlp.fc_2.bias.data = mlp.fc_2.bias.data[mlp_order]
    else:
        raise ValueError("MLP not supported")
    mlp.proj.weight.data = mlp.proj.weight.data[:, mlp_order][embedding_order, :]
    if mlp.proj.bias is not None:
        mlp.proj.bias.data = mlp.proj.bias.data[embedding_order]


def permute_attention(embedding_order, indices_attn, indices_proj, attn):
    attn.proj.weight.data = attn.proj.weight.data[embedding_order, :][:, indices_proj]
    if attn.proj.bias is not None:
        attn.proj.bias.data = attn.proj.bias.data[embedding_order]
    attn.attn.weight.data = attn.attn.weight.data[:, embedding_order][indices_attn, :]
    if attn.attn.bias is not None:
        attn.attn.bias.data = attn.attn.bias.data[indices_attn]


def get_indices_pythia(head_order, n_head, head_size):
    indices_attn = []
    for h in head_order:
        start = 3 * h * head_size
        end = 3 * (h + 1) * head_size
        indices_attn.extend(range(start, end))
    return indices_attn


def permute_model(embedding_order, head_order, mlp_order, model, name):
    model.transformer.wte.weight.data = model.transformer.wte.weight.data[
        :, embedding_order
    ]
    model.lm_head.weight.data = model.lm_head.weight.data[:, embedding_order]
    permute_norm(embedding_order, model.transformer.ln_f)
    if "pythia" in name:
        shape = (
            model.config.n_head + 2 * model.config.n_query_groups
        ) * model.config.head_size
        indices_attn = get_indices_pythia(
            head_order, model.config.n_head, model.config.head_size
        )
        indices_proj = [
            i
            for h in head_order
            for i in range(h * model.config.head_size, (h + 1) * model.config.head_size)
        ]
    else:
        head_to_group = {}
        heads_per_group = model.config.n_head // model.config.n_query_groups
        for id in range(len(head_order)):
            for head in range(heads_per_group):
                head_to_group[id * heads_per_group + head] = head_order[id]
        head_size = model.config.head_size
        indices_attn = []
        for ids in head_order:
            start = ids * (heads_per_group + 2) * head_size
            end = (ids + 1) * (heads_per_group + 2) * head_size
            indices_attn.extend(range(start, end))
        indices_proj = []
        for id in head_order:
            start = id * (heads_per_group) * head_size
            end = (id + 1) * (heads_per_group) * head_size
            indices_proj.extend(range(start, end))

    for block in model.transformer.h:
        permute_norm(embedding_order, block.norm_1)
        permute_norm(embedding_order, block.norm_2)
        permute_mlp(embedding_order, mlp_order, block.mlp)
        permute_attention(embedding_order, indices_attn, indices_proj, block.attn)
    return model
