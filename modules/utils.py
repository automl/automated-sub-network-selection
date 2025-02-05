from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm
from whittle.models.gpt.blocks import GptNeoxMLP, GemmaMLP, LLaMAMLP
import pickle
import copy
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from whittle.modules.layernorm import LayerNorm
from whittle.modules.rmsnorm import RMSNorm
from whittle.models.gpt.blocks import GptNeoxMLP, GemmaMLP, LLaMAMLP
import pickle
from datasets import load_from_disk
# Tokenize each example with dynamic sequence length
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
from pathlib import Path
import shutil
from litgpt import LLM
import os

# Cache Reset and Management
def reset_huggingface_cache(new_cache_dir=None):
    if new_cache_dir:
        new_cache_dir = Path(new_cache_dir)
        os.environ['HF_HOME'] = str(new_cache_dir)
        os.environ['HUGGINGFACE_CACHE'] = str(new_cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(new_cache_dir / "transformers")
        os.environ['HF_DATASETS_CACHE'] = str(new_cache_dir / "datasets")

        print(f"Cache directory set to: {new_cache_dir}")

# Dataset Loading and Sampling
DATASETS = [
    "allenai/c4",
    "wikitext",
    "openwebtext",
    "tatsu-lab/alpaca",
    "zwhe99/commonsense_170k",
    "roneneldan/TinyStories",
    "openai/gsm8k",
]

# Additional datasets you might want to consider
ADDITIONAL_DATASETS = [
    "craffel/openai_lambada",
    "xsum",
    "abisee/cnn_dailymail",
    "Samsung/samsum",
    "yelp/yelp_review_full",
]

DATASETS.extend(ADDITIONAL_DATASETS)
#DATASETS.extend(ADDITIONAL_DATASETS)
def map_to_text_xsum(example):
    example['text'] = f"Document: {example['document']}\nSummary: {example['summary']}"
    return example

def sample_from_dataset(dataset_name, n=100):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    elif dataset_name == "allenai/c4":
        dataset = load_dataset("allenai/c4", "en", split="train")
    elif dataset_name == "craffel/openai_lambada":
        dataset = load_dataset("craffel/openai_lambada", split="test")
    elif dataset_name == "abisee/cnn_dailymail":
        dataset = load_dataset("cnn_dailymail","3.0.0", split="train")
    elif dataset_name == "openai/gsm8k":
        dataset = load_dataset("gsm8k", "main", split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    sampled_data = dataset.shuffle(seed=42).select(range(min(n, len(dataset))))
    return sampled_data

def construct_mixed_dataset(n=100, datasets=DATASETS):
    sampled_datasets = []
    for dataset_name in datasets:
        try:
            sampled = sample_from_dataset(dataset_name, n)

            # Remove label columns if they exist
            if 'label' in sampled.features:
                sampled = sampled.remove_columns(['label'])

            # Map fields for consistency
            if "text" not in sampled.column_names:
                if "question" in sampled.column_names and "context" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Question: {x['question']}\nContext: {x['context']}"})
                elif "title" in sampled.column_names and "content" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": x['content']})
                elif "instruction" in sampled.column_names and "output" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Instruction: {x['instruction']}\nOutput: {x['output']}"})
                elif "document" in sampled.column_names and "summary" in sampled.column_names:
                    sampled = sampled.map(map_to_text_xsum)
                elif "problem" in sampled.column_names and "solution" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Problem: {x['problem']}\nSolution: {x['solution']}"})
                elif "message_1" in sampled.column_names and "message_2" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Message 1: {x['message_1']}\nMessage 2: {x['message_2']}"})
                elif "Problem" in sampled.column_names and "Rationale" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Problem: {x['Problem']}\nRationale: {x['Rationale']}"})
                elif "question" in sampled.column_names and "answer" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"})
                elif "article" in sampled.column_names and "highlights" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Article: {x['article']}\nHighlights: {x['highlights']}"})
                elif "dialogue" in sampled.column_names and "summary" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Dialogue: {x['dialogue']}\nSummary: {x['summary']}"})
                elif "prompt" in sampled.column_names and "response" in sampled.column_names:
                    sampled = sampled.map(lambda x: {"text": f"Prompt: {x['prompt']}\nResponse: {x['response']}"})
                

            sampled_datasets.append(sampled)
            print(f"Sampled {len(sampled)} from {dataset_name}")
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")

    mixed_dataset = concatenate_datasets(sampled_datasets)
    return mixed_dataset

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }
    
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
# Tokenization and DataLoader Creation
def get_dataloader(tokenizer, mixed_dataset, seq_len=512, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token else tokenizer.eos_token_id
        )

    # Apply tokenization
    tokenized_data = tokenize_function(mixed_dataset, tokenizer, seq_len)
    dataset = TextDataset(tokenized_data["input_ids"],tokenized_data["labels"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def evaluate_wikitext(max_length, model, tokenizer, batch_size, num_batches):
    # load mixed dataset
    mixed_dataset = load_from_disk("mixed_dataset/")
    dataloader = get_dataloader(tokenizer, mixed_dataset, max_length, batch_size)
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
            outputs = model(input_ids)
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


def permute_model(embedding_order, head_order, mlp_order, model_base, name):
    model = copy.deepcopy(model_base)
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
        heads_per_group = model.config.n_head//model.config.n_query_groups
        for id in range(len(head_order)):
            for head in range(heads_per_group):
                head_to_group[id*heads_per_group+head] = head_order[id]
        head_size = model.config.head_size
        indices_attn = []
        for ids in head_order:
            start = ids*(heads_per_group+2)*head_size
            end = (ids+1)*(heads_per_group+2)*head_size
            indices_attn.extend(range(start, end))
        indices_proj = []
        for id in head_order:
            start = id*(heads_per_group)*head_size
            end = (id+1)*(heads_per_group)*head_size
            indices_proj.extend(range(start, end))

    for block in model.transformer.h:
        permute_norm(embedding_order, block.norm_1)
        permute_norm(embedding_order, block.norm_2)
        permute_mlp(embedding_order, mlp_order, block.mlp)
        permute_attention(
            embedding_order, indices_attn, indices_proj, block.attn
        )
    return model
