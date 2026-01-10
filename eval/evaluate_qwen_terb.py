import torch
import torch.nn as nn
import numpy as np
import math
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt
from tqdm import tqdm
from collections import defaultdict
import argparse


def ndcg_at_k(relevances, k):
    """Calculate NDCG@k for a single query."""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    
    # DCG
    dcg = np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))
    
    # Ideal DCG
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum((2**ideal_relevances - 1) / np.log2(np.arange(2, len(ideal_relevances) + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def load_model(model_name, use_flash_attention=False):
    """Load the Qwen reranker model using vLLM for multi-GPU tensor parallelism."""
    print(f"Loading model: {model_name}")
    
    # Convert relative paths to absolute paths and check if it's a local path
    import os
    is_local_path = False
    if model_name.startswith('./') or model_name.startswith('../') or os.path.exists(model_name):
        model_name = os.path.abspath(model_name)
        is_local_path = True
        print(f"Resolved to absolute path: {model_name}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Add local_files_only for local paths to avoid HuggingFace Hub validation
    tokenizer_kwargs = {"trust_remote_code": True, "padding_side": "left"}
    if is_local_path:
        tokenizer_kwargs["local_files_only"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    
    # Use vLLM for efficient multi-GPU inference with tensor parallelism
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        max_model_len=8192,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    print(f"Model loaded with vLLM tensor_parallel_size={num_gpus}")
    
    return model, tokenizer


def format_instruction(instruction, query, doc):
    """Format query-document pair with instruction for Qwen3-Reranker."""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )
    return output


def get_reranker_scores_batched(model, tokenizer, query_doc_pairs, batch_size=32, max_length=8000):
    """Get reranker scores for multiple query-document pairs using Qwen3-Reranker with vLLM."""
    
    # Get token IDs for yes/no
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    
    # Suffix for thinking format
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    
    # Sampling params for reranking
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    
    # Create a HF dataset for parallel processing
    from datasets import Dataset as HFDataset
    pairs_ds = HFDataset.from_dict({
        "query": [p[0] for p in query_doc_pairs],
        "doc": [p[1] for p in query_doc_pairs]
    })
    
    def prepare_message(example):
        messages = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\""},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {example['query']}\n\n<Document>: {example['doc']}"}
        ]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        tokens = tokens[:max_length - len(suffix_tokens)] + suffix_tokens
        return {"tokens": tokens}
    
    print("Preparing inputs with parallel processing...")
    pairs_ds = pairs_ds.map(prepare_message, num_proc=96, desc="Tokenizing")
    
    # Create prompts from tokenized messages
    prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in pairs_ds["tokens"]]
    
    # Generate scores using vLLM
    print(f"Running inference on {len(prompts)} pairs...")
    outputs = model.generate(prompts, sampling_params, use_tqdm=True)
    
    # Extract scores
    scores = []
    for output in outputs:
        final_logits = output.outputs[0].logprobs[-1]
        
        if true_token not in final_logits:
            true_logit = -10
        else:
            true_logit = final_logits[true_token].logprob
        
        if false_token not in final_logits:
            false_logit = -10
        else:
            false_logit = final_logits[false_token].logprob
        
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        score = true_score / (true_score + false_score)
        scores.append(score)
    
    return scores


def evaluate_model(model_name, dataset, batch_size=32, use_flash_attention=False):
    """Evaluate a single model on the dataset."""
    model, tokenizer = load_model(model_name, use_flash_attention)
    
    # Group data by query
    query_data = defaultdict(list)
    for row in tqdm(dataset, desc="Grouping by query"):
        query = row["query"]
        if row['description']:
            document = f"Title: {row['title']}\n\nDescription: {row['description']}"
        else:
            document = f"Title: {row['title']}"
        relevance = row["relevance_score"]
        query_data[query].append({"document": document, "relevance": relevance})
    
    # Flatten all query-document pairs for batched processing
    all_pairs = []
    pair_to_query_idx = []
    query_list = list(query_data.keys())
    
    for query_idx, query in enumerate(query_list):
        items = query_data[query]
        for item in items:
            all_pairs.append((query, item["document"]))
            pair_to_query_idx.append(query_idx)
    
    print(f"Total query-document pairs: {len(all_pairs)}")
    
    # Get all scores in batched manner
    all_scores = get_reranker_scores_batched(model, tokenizer, all_pairs, batch_size)
    
    # Reconstruct per-query results
    query_scores = defaultdict(list)
    query_relevances = defaultdict(list)
    
    for idx, (score, query_idx) in enumerate(zip(all_scores, pair_to_query_idx)):
        query = query_list[query_idx]
        query_scores[query].append(score)
        query_relevances[query].append(query_data[query][len(query_scores[query]) - 1]["relevance"])
    
    # Calculate NDCG for each query
    ndcg_5_scores = []
    ndcg_10_scores = []
    query_results = []
    
    for query in tqdm(query_list, desc="Calculating NDCG"):
        scores = query_scores[query]
        ground_truth = query_relevances[query]
        
        # Sort by reranker scores (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_relevances = [ground_truth[i] for i in sorted_indices]
        
        # Calculate NDCG
        ndcg_5 = ndcg_at_k(sorted_relevances, k=5)
        ndcg_10 = ndcg_at_k(sorted_relevances, k=10)
        
        ndcg_5_scores.append(ndcg_5)
        ndcg_10_scores.append(ndcg_10)
        
        query_results.append({
            "query": query,
            "ndcg_5": ndcg_5,
            "ndcg_10": ndcg_10,
            "num_documents": len(scores)
        })
    
    # Clean up
    from vllm.distributed.parallel_state import destroy_model_parallel
    import gc
    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "ndcg@5": np.mean(ndcg_5_scores),
        "ndcg@10": np.mean(ndcg_10_scores),
        "num_queries": len(query_data),
        "query_results": query_results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen Reranker models")
    parser.add_argument("--data_path", type=str, default="filtered_reranker_data.hf", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--models", nargs="+", default=["Qwen3-Reranker-0.6B", "Qwen3-Reranker-4B"], 
                        help="List of model names to evaluate")
    parser.add_argument("--flash_attention", action="store_true", help="Enable Flash Attention 2")
    args = parser.parse_args()
    
    # Default Qwen reranker models
    models = {
        "Qwen-finetuned-hf": "thebajajra/RexReranker-0.6B",
        # "Qwen3-Reranker-0.6B": "Qwen/Qwen3-Reranker-0.6B",
        # "Qwen3-Reranker-4B": "Qwen/Qwen3-Reranker-4B",
        # "Qwen3-Reranker-8B": "Qwen/Qwen3-Reranker-8B",
    }
    
    print(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)['test']
    print(f"Dataset size: {len(dataset)}")
    
    results = {}
    
    for model_label, model_name in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_label}")
        print(f"{'='*50}")
        
        result = evaluate_model(model_name, dataset, args.batch_size, args.flash_attention)
        results[model_label] = result
        
        print(f"\nResults for {model_label}:")
        print(f"  NDCG@5:  {result['ndcg@5']:.4f}")
        print(f"  NDCG@10: {result['ndcg@10']:.4f}")
        print(f"  Number of queries: {result['num_queries']}")
        
        # Save all query results
        query_results = result["query_results"]
        
        # Save as HF dataset
        ds_results = Dataset.from_list(query_results)
        output_path = f"query_ndcg_results_{model_label.replace('/', '_')}_{args.data_path.split('/')[-1]}"
        ds_results.save_to_disk(output_path)
        print(f"  Saved {len(query_results)} query results to {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<25} {'NDCG@5':<10} {'NDCG@10':<10}")
    print("-" * 45)
    for model_label, result in results.items():
        print(f"{model_label:<25} {result['ndcg@5']:<10.4f} {result['ndcg@10']:<10.4f}")


if __name__ == "__main__":
    main()