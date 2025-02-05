### Running importance scoring followed by search

```
python importance/get_importance.py --model_id EleutherAI/pythia-1b --num_batches 100 --objective norm
python llm_compression/fine_tuning EleutherAI/pythia-1b --sampling_strategy "importance-grid-params" --importance_objective norm
```