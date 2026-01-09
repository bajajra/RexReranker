import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


def load_model(model_name):
    """Load the NVIDIA Nemotron reranker model."""
    print(f"Loading model: {model_name}")
    
    import os
    if model_name.startswith('./') or model_name.startswith('../') or os.path.exists(model_name):
        model_name = os.path.abspath(model_name)
        print(f"Resolved to absolute path: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with automatic device mapping for multi-GPU
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    
    num_gpus = torch.cuda.device_count()
    print(f"Model loaded on {num_gpus} GPU(s)")
    
    return model, tokenizer


def get_reranker_scores_batched(model, tokenizer, query_doc_pairs, batch_size=32, max_length=4096):
    """Get reranker scores for multiple query-document pairs using NVIDIA Nemotron reranker."""
    
    all_scores = []
    
    # Process in batches
    for i in tqdm(range(0, len(query_doc_pairs), batch_size), desc="Scoring batches"):
        batch_pairs = query_doc_pairs[i:i + batch_size]
        
        queries = [p[0] for p in batch_pairs]
        documents = [p[1] for p in batch_pairs]
        
        # Tokenize query-document pairs
        inputs = tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get scores
        with torch.no_grad():
            outputs = model(**inputs)
            # The model outputs logits - higher means more relevant
            scores = outputs.logits.squeeze(-1).float().cpu().numpy()
        
        # Handle single item batches
        if scores.ndim == 0:
            scores = [float(scores)]
        else:
            scores = scores.tolist()
        
        all_scores.extend(scores)
    
    return all_scores


def evaluate_model(model_name, dataset, batch_size=32):
    """Evaluate a single model on the dataset."""
    model, tokenizer = load_model(model_name)
    
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
    import gc
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
    parser = argparse.ArgumentParser(description="Evaluate NVIDIA Nemotron Reranker model")
    parser.add_argument("--data_path", type=str, default="filtered_reranker_data.hf", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--model", type=str, default="nvidia/llama-nemotron-rerank-1b-v2",
                        help="Model name or path")
    args = parser.parse_args()
    
    models = {
        "Nemotron-Rerank-1B": args.model,
    }
    
    print(f"Loading dataset from {args.data_path}")
    dataset = load_from_disk(args.data_path)['test']
    print(f"Dataset size: {len(dataset)}")
    
    results = {}
    
    for model_label, model_name in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {model_label}")
        print(f"{'='*50}")
        
        result = evaluate_model(model_name, dataset, args.batch_size)
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

