#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from packaging import version
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Features, Value, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_main_process() -> bool:
    """Check if current process is main (rank 0) for distributed training."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def log_info(msg: str):
    """Log info only on main process."""
    if is_main_process():
        logger.info(msg)


# ----------------------------
# Prompt formatting (aligned with Qwen3 reranker-style usage)
# ----------------------------
DEFAULT_PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n'
    '<|im_start|>user\n'
)
DEFAULT_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)

def format_instruction(instruction: str, query: str, title: str, description: Optional[str] = None) -> str:
    """Format the input for the reranker model.
    
    Args:
        instruction: Task instruction for the model.
        query: The search query.
        title: Product title.
        description: Optional product description.
    
    Returns:
        Formatted string for model input.
    """
    if description and description.strip():
        document = f"{title}\n{description}"
    else:
        document = title
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


# ----------------------------
# Model: backbone + distribution head
# ----------------------------
class Qwen3DistributionReranker(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_bins: int = 11,
        dropout: float = 0.1,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ):
        super().__init__()

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype)

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )

        if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone config.")

        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, num_bins)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SequenceClassifierOutput:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]

        # pool last non-pad token (works with left or right padding)
        idx = attention_mask.long().sum(dim=1) - 1  # [B]
        batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch, idx, :]  # [B, H]

        logits = self.score_head(self.dropout(pooled))  # [B, K]
        return SequenceClassifierOutput(logits=logits)


# ----------------------------
# Per-transition dynamic sigma (0.8 narrower)
# ----------------------------
def sigma_from_transitions_nearest(
    y: torch.Tensor,                 # [B] in [0,1]
    transitions: List[float],        # [M]
    sigma_min: float,                # scalar
    sigma_maxes: List[float],        # [M] peak sigma per transition
    deltas: List[float],             # [M] width per transition
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Picks nearest transition per example, then applies a Gaussian bump with
    that transition's sigma_max and delta.

    sigma(y) = sigma_min + (sigma_max_j - sigma_min) * exp(-0.5 * (d/delta_j)^2)
    where j is nearest boundary index, d = |y - t_j|
    """
    y = torch.clamp(y, 0.0, 1.0)
    device = y.device
    dtype = y.dtype

    t = torch.tensor(transitions, device=device, dtype=dtype).view(1, -1)  # [1, M]
    diffs = torch.abs(y.view(-1, 1) - t)  # [B, M]
    d, j = torch.min(diffs, dim=1)        # [B], [B] index

    sigma_max_t = torch.tensor(sigma_maxes, device=device, dtype=dtype)  # [M]
    delta_t = torch.tensor(deltas, device=device, dtype=dtype)           # [M]

    sigma_max_j = sigma_max_t[j]  # [B]
    delta_j = delta_t[j]          # [B]

    closeness = torch.exp(-0.5 * (d * d) / (delta_j * delta_j + eps))    # [B]
    sigma = sigma_min + (sigma_max_j - sigma_min) * closeness
    return torch.clamp(sigma, min=eps)


# ----------------------------
# Truncated (by discretization) + renormalized Gaussian on bin centers
# ----------------------------
def gaussian_soft_targets(
    labels: torch.Tensor,      # [B]
    bin_centers: torch.Tensor, # [K]
    sigma: torch.Tensor,       # [B]
    eps: float = 1e-8,
) -> torch.Tensor:
    labels = labels.view(-1, 1)       # [B, 1]
    centers = bin_centers.view(1, -1) # [1, K]
    sigma2 = (sigma.view(-1, 1) ** 2) # [B, 1]

    w = torch.exp(-0.5 * (labels - centers) ** 2 / (sigma2 + eps))
    w = torch.clamp(w, min=eps)
    w = w / w.sum(dim=1, keepdim=True)
    return w


# ----------------------------
# Freeze/unfreeze callback
# ----------------------------
class UnfreezeBackboneAfterFirstEpochCallback(TrainerCallback):
    """Callback to unfreeze the backbone after a specified number of epochs.
    
    This is useful for staged training where we first train only the head,
    then fine-tune the entire model.
    """
    def __init__(self, unfreeze_epoch: int = 1):
        self.unfreeze_epoch = unfreeze_epoch
        self.did_unfreeze = False

    def _get_unwrapped_model(self, model):
        """Get the underlying model from DDP/FSDP wrapper."""
        if hasattr(model, "module"):
            return model.module
        return model

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if self.did_unfreeze:
            return control
        if state.epoch is not None and state.epoch >= self.unfreeze_epoch:
            # Get the unwrapped model (handles DDP/FSDP wrappers)
            unwrapped = self._get_unwrapped_model(model)
            # Unfreeze backbone parameters
            for p in unwrapped.backbone.parameters():
                p.requires_grad = True
            self.did_unfreeze = True
            log_info(f"[callback] Unfroze backbone at epoch={state.epoch}")
        return control


# ----------------------------
# Custom Trainer: KL + lambda * mean MSE, per-transition sigma
# ----------------------------
class DistributionHybridTrainer(Trainer):
    def __init__(
        self,
        *args,
        bin_centers: torch.Tensor,
        transitions: List[float],
        sigma_min: float,
        sigma_maxes: List[float],
        sigma_deltas: List[float],
        lambda_mean: float,
        head_lr: float,
        backbone_lr: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._bin_centers = bin_centers
        self._transitions = transitions
        self._sigma_min = sigma_min
        self._sigma_maxes = sigma_maxes
        self._sigma_deltas = sigma_deltas
        self._lambda_mean = lambda_mean
        self._head_lr = head_lr
        self._backbone_lr = backbone_lr

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        # Remove query_group_id if present (used for metrics only)
        inputs.pop("query_group_id", None)
        outputs = model(**inputs)
        logits = outputs.logits  # [B, K]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        y = torch.clamp(labels.to(log_probs.device), 0.0, 1.0)

        # per-transition sigma (0.8 narrower via sigma_maxes/deltas)
        sigma = sigma_from_transitions_nearest(
            y=y,
            transitions=self._transitions,
            sigma_min=self._sigma_min,
            sigma_maxes=self._sigma_maxes,
            deltas=self._sigma_deltas,
        )

        target = gaussian_soft_targets(
            labels=y,
            bin_centers=self._bin_centers.to(log_probs.device),
            sigma=sigma,
        )

        kl_loss = F.kl_div(log_probs, target, reduction="batchmean")

        bin_centers = self._bin_centers.to(probs.device)
        pred_mean = (probs * bin_centers.view(1, -1)).sum(dim=1)
        mean_mse = F.mse_loss(pred_mean, y)

        loss = kl_loss + self._lambda_mean * mean_mse

        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle query_group_id in inputs."""
        # Remove query_group_id before prediction (it's metadata, not model input)
        inputs.pop("query_group_id", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        def is_no_decay(name: str) -> bool:
            lname = name.lower()
            if lname.endswith("bias"):
                return True
            if "norm" in lname and lname.endswith("weight"):
                return True
            return False

        head_decay, head_no_decay = [], []
        bb_decay, bb_no_decay = [], []

        for name, p in self.model.named_parameters():
            if p is None:
                continue
            if name.startswith("score_head."):
                (head_no_decay if is_no_decay(name) else head_decay).append(p)
            else:
                (bb_no_decay if is_no_decay(name) else bb_decay).append(p)

        optimizer_grouped_parameters = [
            {"params": bb_decay, "weight_decay": self.args.weight_decay, "lr": self._backbone_lr},
            {"params": bb_no_decay, "weight_decay": 0.0, "lr": self._backbone_lr},
            {"params": head_decay, "weight_decay": self.args.weight_decay, "lr": self._head_lr},
            {"params": head_no_decay, "weight_decay": 0.0, "lr": self._head_lr},
        ]

        fused_ok = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_ok and torch.cuda.is_available()

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            fused=use_fused,
        )
        return self.optimizer


# ----------------------------
# Metrics
# ----------------------------
def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute DCG@k for a single query's ranked list."""
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    # DCG = sum_{i=1}^{k} (2^rel_i - 1) / log2(i + 1)
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    gains = (2 ** relevances - 1) / discounts
    return float(np.sum(gains))


def ndcg_at_k(true_relevances: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single query.
    
    Args:
        true_relevances: Ground truth relevance scores.
        predicted_scores: Model predicted scores for ranking.
        k: Cutoff for NDCG calculation.
    
    Returns:
        NDCG@k score in [0, 1].
    """
    if len(true_relevances) == 0:
        return 0.0
    
    # Rank by predicted scores (descending)
    ranked_indices = np.argsort(-predicted_scores)
    ranked_relevances = true_relevances[ranked_indices]
    
    # Compute DCG@k
    dcg = dcg_at_k(ranked_relevances, k)
    
    # Compute ideal DCG@k (sort by true relevance descending)
    ideal_relevances = np.sort(true_relevances)[::-1]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_ndcg_at_k_grouped(
    true_relevances: np.ndarray,
    predicted_scores: np.ndarray,
    query_ids: np.ndarray,
    k_values: List[int],
) -> Dict[str, float]:
    """Compute NDCG@k for multiple k values, grouped by query.
    
    Args:
        true_relevances: Array of ground truth relevance scores.
        predicted_scores: Array of model predicted scores.
        query_ids: Array of query identifiers for grouping.
        k_values: List of k values for NDCG@k computation.
    
    Returns:
        Dictionary with NDCG@k for each k value.
    """
    unique_queries = np.unique(query_ids)
    
    # Initialize accumulators for each k
    ndcg_sums = {k: 0.0 for k in k_values}
    valid_query_counts = {k: 0 for k in k_values}
    
    for qid in unique_queries:
        mask = query_ids == qid
        q_relevances = true_relevances[mask]
        q_scores = predicted_scores[mask]
        
        # Skip queries with no relevant items (all zeros)
        if np.sum(q_relevances) == 0:
            continue
        
        for k in k_values:
            ndcg = ndcg_at_k(q_relevances, q_scores, k)
            ndcg_sums[k] += ndcg
            valid_query_counts[k] += 1
    
    results = {}
    for k in k_values:
        if valid_query_counts[k] > 0:
            results[f"ndcg@{k}"] = ndcg_sums[k] / valid_query_counts[k]
        else:
            results[f"ndcg@{k}"] = 0.0
    
    return results


def compute_metrics_builder(num_bins: int, query_ids: Optional[np.ndarray] = None, ndcg_k_values: Optional[List[int]] = None):
    """Build metrics computation function.
    
    Args:
        num_bins: Number of distribution bins.
        query_ids: Optional array of query IDs for NDCG computation.
        ndcg_k_values: Optional list of k values for NDCG@k (e.g., [5, 10, 20]).
    
    Returns:
        Metrics computation function.
    """
    bin_centers = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids.astype(np.float32)
        probs = softmax_np(logits, axis=-1)
        
        # Scalar score = expected value = sum(probs * bin_centers)
        pred_mean = (probs * bin_centers[None, :]).sum(axis=1)

        # Basic regression metrics
        mse = float(np.mean((pred_mean - labels) ** 2))
        mae = float(np.mean(np.abs(pred_mean - labels)))
        pred_var = float(np.mean((probs * (bin_centers[None, :] - pred_mean[:, None]) ** 2).sum(axis=1)))
        entropy = float(np.mean(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1)))
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "mean_pred_variance": pred_var,
            "mean_entropy": entropy,
        }
        
        # Compute NDCG@k if query_ids are provided
        if query_ids is not None and ndcg_k_values is not None and len(ndcg_k_values) > 0:
            ndcg_metrics = compute_ndcg_at_k_grouped(
                true_relevances=labels,
                predicted_scores=pred_mean,
                query_ids=query_ids,
                k_values=ndcg_k_values,
            )
            metrics.update(ndcg_metrics)
        
        return metrics

    return compute_metrics


def parse_floats_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    
    # Dataset arguments - either HF hub or local files
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name (e.g., 'my-org/my-dataset')")
    parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration name")
    parser.add_argument("--train_file", type=str, default=None, help="Path to local training file (JSON)")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to local validation file (JSON)")
    parser.add_argument("--validation_split", action="store_true", help="Create validation split from train if not provided")
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--instruction", type=str, default="Given an e-commerce search query, judge whether the product details satisfy the query.")
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    parser.add_argument("--suffix", type=str, default=DEFAULT_SUFFIX)

    parser.add_argument("--num_bins", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Per-transition sigma controls
    parser.add_argument("--transitions", type=str, default="0.2,0.5,0.8")
    parser.add_argument("--sigma_min", type=float, default=0.04)
    # narrower at 0.8 by default:
    parser.add_argument("--sigma_maxes", type=str, default="0.12,0.12,0.08")
    # optional: tighter influence near 0.8
    parser.add_argument("--sigma_deltas", type=str, default="0.08,0.08,0.06")

    # Hybrid loss
    parser.add_argument("--lambda_mean", type=float, default=10.0)
    
    # NDCG evaluation
    parser.add_argument("--ndcg_k", type=str, default="5,10,20", help="Comma-separated k values for NDCG@k (e.g., '5,10,20')")

    # LRs
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)

    # Training setup
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--attn_implementation", type=str, default=None)  # e.g. "flash_attention_2"
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (set by torchrun)")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file")
    parser.add_argument("--ddp_timeout", type=int, default=1800, help="DDP timeout in seconds")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()

    # Qwen3 support guard
    if version.parse(transformers.__version__) < version.parse("4.51.0"):
        raise RuntimeError(
            f"transformers=={transformers.__version__} is too old for Qwen3. "
            "Install transformers>=4.51.0."
        )

    # Validate dataset arguments
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Either --dataset_name or --train_file must be provided.")
    if args.dataset_name is not None and args.train_file is not None:
        raise ValueError("Provide either --dataset_name or --train_file, not both.")

    transitions = parse_floats_csv(args.transitions)
    sigma_maxes = parse_floats_csv(args.sigma_maxes)
    sigma_deltas = parse_floats_csv(args.sigma_deltas)

    assert len(transitions) == len(sigma_maxes) == len(sigma_deltas), \
        "transitions, sigma_maxes, sigma_deltas must have the same length."
    assert all(0.0 <= t <= 1.0 for t in transitions), "All transitions must be in [0,1]."
    assert args.sigma_min > 0.0, "sigma_min must be > 0."
    assert all(sm >= args.sigma_min for sm in sigma_maxes), "each sigma_max must be >= sigma_min."
    assert all(d > 0.0 for d in sigma_deltas), "each sigma_delta must be > 0."

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefix_tokens = tokenizer.encode(args.prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(args.suffix, add_special_tokens=False)

    # Load dataset - supports both local files and HuggingFace Hub datasets
    if args.dataset_name:
        log_info(f"Loading dataset from HuggingFace Hub: {args.dataset_name}")
        ds = load_dataset(args.dataset_name, args.dataset_config)
        
        # Convert to DatasetDict if it's a single Dataset
        if not isinstance(ds, DatasetDict):
            ds = DatasetDict({"train": ds})
        
        if args.validation_split and "validation" not in ds:
            # Split train into train/validation if validation not present
            log_info("Creating validation split from training data (10%)")
            split_ds = ds["train"].train_test_split(test_size=0.1, seed=args.seed)
            ds = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})
    else:
        log_info(f"Loading dataset from local files")
        data_files = {"train": args.train_file}
        if args.validation_file:
            data_files["validation"] = args.validation_file
        ds = load_dataset("json", data_files=data_files)

    # Expected columns: query, title, description, relevance_score
    expected_cols = {"query", "title", "relevance_score"}
    train_cols = set(ds["train"].column_names)
    missing = expected_cols - train_cols
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {train_cols}")

    # Cast to expected types
    features = Features({
        "query": Value("string"),
        "title": Value("string"),
        "description": Value("string"),
        "relevance_score": Value("float32"),
    })
    
    # Handle optional description column
    def ensure_description(example):
        if "description" not in example or example["description"] is None:
            example["description"] = ""
        return example
    
    ds = ds.map(ensure_description)
    
    # Only cast columns that exist
    cast_features = {k: v for k, v in features.items() if k in ds["train"].column_names}
    ds = ds.cast(Features(cast_features))

    # Create query_id mapping for NDCG calculation
    # We need to assign numeric IDs to queries for grouping during evaluation
    log_info("Creating query ID mappings for NDCG evaluation...")
    
    def add_query_id(example, idx):
        example["query_id"] = idx
        example["query_text"] = example["query"]  # Keep original query for grouping
        return example
    
    ds = ds.map(add_query_id, with_indices=True)
    
    # Build query -> id mapping from validation set for consistent grouping
    query_to_id: Dict[str, int] = {}
    
    def assign_query_group_id(example):
        query = example["query_text"]
        if query not in query_to_id:
            query_to_id[query] = len(query_to_id)
        example["query_group_id"] = query_to_id[query]
        return example
    
    # Process validation first to build the mapping, then train
    if "validation" in ds:
        ds["validation"] = ds["validation"].map(assign_query_group_id)
    ds["train"] = ds["train"].map(assign_query_group_id)

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        texts = [
            format_instruction(args.instruction, q, t, d)
            for q, t, d in zip(batch["query"], batch["title"], batch["description"])
        ]

        mid_max_len = args.max_length - len(prefix_tokens) - len(suffix_tokens)
        if mid_max_len <= 16:
            raise ValueError("max_length too small after accounting for prefix/suffix tokens.")

        enc = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=mid_max_len,
            padding=False,
            return_attention_mask=False,
        )

        input_ids = [prefix_tokens + ids + suffix_tokens for ids in enc["input_ids"]]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": batch["relevance_score"],
            "query_group_id": batch["query_group_id"],  # Keep for NDCG grouping
        }
        return result

    # Remove all columns except the ones we need
    cols_to_keep = {"input_ids", "attention_mask", "labels", "query_group_id"}
    cols_to_remove = [c for c in ds["train"].column_names if c not in cols_to_keep]
    ds = ds.map(preprocess_batch, batched=True, remove_columns=cols_to_remove)

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Remove query_group_id before padding (it's only for metrics, not model input)
        query_group_ids = [f.pop("query_group_id", None) for f in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        # Add query_group_id back for metrics computation (won't be passed to model)
        if query_group_ids[0] is not None:
            batch["query_group_id"] = torch.tensor(query_group_ids, dtype=torch.long)
        return batch

    model = Qwen3DistributionReranker(
        backbone_name=args.model_name_or_path,
        num_bins=args.num_bins,
        dropout=args.dropout,
        attn_implementation=args.attn_implementation,
        torch_dtype=("bfloat16" if args.bf16 else ("float16" if args.fp16 else None)),
    )

    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    # Freeze backbone for epoch 0
    for p in model.backbone.parameters():
        p.requires_grad = False

    bin_centers = torch.linspace(0.0, 1.0, args.num_bins, dtype=torch.float32)

    # Determine if we have validation data
    has_validation = args.validation_file is not None or (args.dataset_name and args.validation_split)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.backbone_lr,  # Base LR for scheduler (we use custom optimizer with separate LRs)
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if has_validation else "no",
        eval_steps=args.eval_steps if has_validation else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        # Multi-GPU / Distributed training settings
        ddp_find_unused_parameters=True,  # Required because we freeze backbone initially
        ddp_timeout=args.ddp_timeout,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        deepspeed=args.deepspeed,
        local_rank=args.local_rank,
    )

    callbacks = [UnfreezeBackboneAfterFirstEpochCallback(unfreeze_epoch=1)]

    # Get validation dataset if available
    eval_dataset = None
    eval_query_ids = None
    if "validation" in ds:
        eval_dataset = ds["validation"]
        # Extract query_group_ids for NDCG computation
        eval_query_ids = np.array(eval_dataset["query_group_id"])
        log_info(f"Validation set has {len(np.unique(eval_query_ids))} unique queries")
    
    # Parse NDCG k values
    ndcg_k_values = [int(k.strip()) for k in args.ndcg_k.split(",") if k.strip()]
    log_info(f"NDCG will be computed at k={ndcg_k_values}")
    
    # Build metrics function with query IDs for NDCG
    compute_metrics_fn = None
    if eval_dataset:
        compute_metrics_fn = compute_metrics_builder(
            num_bins=args.num_bins,
            query_ids=eval_query_ids,
            ndcg_k_values=ndcg_k_values,
        )
    
    trainer = DistributionHybridTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        data_collator=collate,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,

        bin_centers=bin_centers,
        transitions=transitions,
        sigma_min=args.sigma_min,
        sigma_maxes=sigma_maxes,
        sigma_deltas=sigma_deltas,
        lambda_mean=args.lambda_mean,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
    )

    log_info(f"Starting training with {len(ds['train'])} training examples")
    if eval_dataset:
        log_info(f"Validation set size: {len(eval_dataset)}")
    
    trainer.train()
    
    # Save model (Trainer handles distributed saving internally)
    trainer.save_model(args.output_dir)
    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Training complete.")
        logger.info(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()