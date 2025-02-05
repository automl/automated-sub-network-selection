import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import copy


def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)

    model_copy = deepcopy(model)
    for diff_lay in range(merge_base_lay + 1, merge_base_lay + 1 + merge_layer_num):
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data
            - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data
            - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data
            - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        )

        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data
            - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )

        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data
            - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        )

        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data
            - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )

        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data
            - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )

    for diff_lay in range(merge_base_lay + merge_layer_num, merge_base_lay, -1):
        del model_copy.model.layers[diff_lay]
    return model_copy


from utils import evaluate_wikitext


def evaluate_model(model, tokenizer, batch_size, num_batches):
    return evaluate_wikitext(1024, model, tokenizer, batch_size, num_batches)


def cal_last_hidden_sim(model1, model2, tokenizer, sents):
    sim_ls = []
    model1.to("cuda")
    model2.to("cuda")
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors="pt")
        encoded_inputs["input_ids"] = encoded_inputs["input_ids"].to("cuda")
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].to("cuda")
        with torch.no_grad():
            outputs1 = model1(
                **encoded_inputs, output_hidden_states=True, use_cache=False
            )
        hidden_states1 = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
        with torch.no_grad():
            outputs2 = model2(
                **encoded_inputs, output_hidden_states=True, use_cache=False
            )
        hidden_states2 = outputs2.hidden_states[-1]  # (1, seq_len, hidden)
        sim_ls.append(
            torch.cosine_similarity(
                hidden_states1.squeeze(0).flatten().unsqueeze(0),
                hidden_states2.squeeze(0).flatten().unsqueeze(0),
            )
        )
    sim_ls = [i.item() for i in sim_ls]
    print(sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)


import argparse


parser = argparse.ArgumentParser(
    description="Script for merging layers with specific parameters."
)

# Adding arguments
parser.add_argument(
    "--interval", type=int, default=1, help="The interval for merging layers."
)
parser.add_argument(
    "--merge_layers", type=int, default=8, help="Number of layers to merge."
)
parser.add_argument(
    "--lowest_lay", type=int, default=0, help="The lowest layer index in the model."
)
parser.add_argument(
    "--threshold", type=float, default=0.2, help="Threshold value for the operation."
)
parser.add_argument(
    "--num_batches", type=int, default=1, help="Number of batches for processing."
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="Batch size for processing."
)
parser.add_argument(
    "--model_id", type=str, default="meta-llama/Llama-3.1-8B", help="Type of llama"
)
parser.add_argument("--limit", type=int, default=None)
tasks = "minerva_math,mathqa"
task_to_shot = {
    "winogrande": 5,
    "hellaswag": 10,
    "arc_challenge": 25,
    "piqa": 0,
    "boolq": 0,
    "truthfulqa_mc2": 0,
    "gsm8k": 5,
    "mmlu": 5,
    "lambada_openai": 0,
    "minerva_math": 4,
    "mathqa": 0,
}
metric_map = {
    "winogrande": "acc",
    "arc_challenge": "acc_norm",
    "mmlu": "acc,none",
    "hellaswag": "acc_norm",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc2": "acc,none",
    "boolq": "acc,none",
    "piqa": "acc",
    "lambada_openai": "acc",
    "minerva_math": "exact_match,none",
    "mathqa": "acc_norm,none",
}
tasks = tasks.split(",")
args = parser.parse_args()
en_wiki_selected = [
    "Mouron () is a commune in the Arde",
    "The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force",
    "There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly",
    "Torreorgaz is a municipality in the",
    "Copa Libertadores 1973 was won by defending champions Independiente of A",
]
sents = []
sents.extend(en_wiki_selected)
model_id = args.model_id
llama = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
llama = llama.to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

llama_compressed = copy.deepcopy(llama)
args.highest_lay = len(llama_compressed.model.layers) - 1
# Calculate `lay` and `last_merge_flag` based on arguments
lay = args.highest_lay - args.merge_layers
last_merge_flag = False  # Default, can be updated based on logic

print("Number of layers in model", len(llama_compressed.model.layers))
# Print or use these values
print(f"Interval: {args.interval}")
print(f"Merge Layers: {args.merge_layers}")
print(f"Highest Layer: {args.highest_lay}")
print(f"Lowest Layer: {args.lowest_lay}")
print(f"Threshold: {args.threshold}")
print(f"Number of Batches: {args.num_batches}")
print(f"Batch Size: {args.batch_size}")
print(f"Lay: {lay}")
print(f"Last Merge Flag: {last_merge_flag}")

print(
    "PPL before merging",
    evaluate_model(llama, tokenizer, args.batch_size, args.num_batches),
)
while lay >= args.lowest_lay:
    print(lay)
    print("current model layer", len(llama_compressed.model.layers))
    tmp_merged_model = merge_layers_return_model(
        llama_compressed, lay, args.merge_layers - 1
    )
    sim_value = cal_last_hidden_sim(llama, tmp_merged_model, tokenizer, sents)
    if sim_value > args.threshold:
        llama_compressed = tmp_merged_model
        lay -= args.interval
        if lay >= len(llama_compressed.model.layers):
            lay = len(llama_compressed.model.layers) - 1 - args.merge_layers
    else:
        lay -= 1

# Update number of hidden layers in the compressed model configuration
llama_compressed.config.num_hidden_layers = len(llama_compressed.model.layers)

# Calculate compression ratio (number of layers in original vs. compressed model)
num_params = len(llama.model.layers) / len(llama_compressed.model.layers)

# Evaluate the compressed model (e.g., calculate perplexity on evaluation data)
ppl = evaluate_model(llama_compressed, tokenizer, args.batch_size, args.num_batches)

# Print evaluation results
print("PPL after merging:", ppl)
print("Layers after merging:", llama_compressed.config.num_hidden_layers)

# Save the compressed model with a unique identifier based on `num_params`
model_save_path = (
    f"llama_compressed_{int(llama_compressed.config.num_hidden_layers)}.bin"
)
llama_compressed.save_pretrained(model_save_path)
print(f"Compressed model saved at {model_save_path}")
import lm_eval

all_datasets = {}
all_datasets["num_layers_merged"] = llama_compressed.config.num_hidden_layers
all_datasets["num_layers_base"] = args.highest_lay + 1
import pickle
from lm_eval.models.huggingface import HFLM

for t in tasks:
    lm_obj = HFLM(pretrained=llama_compressed)
    result = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=[t],
        num_fewshot=task_to_shot[t],
        batch_size="auto",
        limit=args.limit,
    )
    all_datasets[t] = result["results"][t][metric_map[t]]
    with open(f"laco_math_{args.merge_layers}_{args.threshold}.pkl", "wb") as f:
        pickle.dump(all_datasets, f)
