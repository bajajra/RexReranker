#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune RexBERT-base encoder as a distributional reranker:
- Predict categorical distribution over K bins in [0, 1]
- Target: truncated (by discretization) + renormalized Gaussian centered at label
- Sigma is dynamic: highest near transition points {0.2, 0.5, 0.8}
- Loss: KL(target || pred) + lambda_mean * MSE(E[pred], y)

Schedule:
- First N epochs: train head only (backbone frozen), where N = unfreeze_backbone_after (default: 0.1)
- After N epochs: unfreeze backbone and fine-tune all layers

Input format: Query: {query} [SEP] Title: {title} \n Description: {description}
Pooling: mean (default) or [CLS] token

Dataset: thebajajra/Amazebay-reranker-training-data (HuggingFace)
"""

import os
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Value


# ----------------------------
# Model: RexBERT backbone + distribution head
# ----------------------------
class RexBERTDistributionReranker(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_bins: int = 11,
        dropout: float = 0.1,
        pooling_strategy: str = "mean",
        torch_dtype: Optional[str] = None,
    ):
        super().__init__()

        assert pooling_strategy in ("cls", "mean"), f"pooling_strategy must be 'cls' or 'mean', got {pooling_strategy}"
        self.pooling_strategy = pooling_strategy

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype)

        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # reduce memory during training
        if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "use_cache"):
            self.backbone.config.use_cache = False

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone config.")

        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, num_bins)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # accepted but unused (handled in Trainer.compute_loss)
    ) -> SequenceClassifierOutput:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]

        if self.pooling_strategy == "cls":
            # [CLS] token pooling (position 0)
            pooled = last_hidden[:, 0, :]  # [B, H]
        else:
            # Mean pooling over non-padding tokens (generally better for retrieval/reranking)
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)     # [B, H]
            lengths = mask.sum(dim=1).clamp(min=1e-9)    # [B, 1]
            pooled = summed / lengths                    # [B, H]

        logits = self.score_head(self.dropout(pooled))  # [B, K]
        return SequenceClassifierOutput(logits=logits)


# ----------------------------
# Dynamic sigma around transition points
# ----------------------------
def sigma_from_transitions(
    y: torch.Tensor,
    transitions: List[float],
    sigma_min: float,
    sigma_max: float,
    delta: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    y: [B] in [0,1]
    transitions: list of boundary points, e.g. [0.2, 0.5, 0.8]
    Returns sigma: [B], where sigma peaks near transitions.
    """
    y = torch.clamp(y, 0.0, 1.0)
    t = torch.tensor(transitions, device=y.device, dtype=y.dtype).view(1, -1)  # [1, M]
    y2 = y.view(-1, 1)  # [B, 1]
    d = torch.min(torch.abs(y2 - t), dim=1).values  # [B] distance to nearest boundary
    # RBF bump: closeness in [0,1]
    closeness = torch.exp(-0.5 * (d * d) / (delta * delta + eps))  # [B]
    sigma = sigma_min + (sigma_max - sigma_min) * closeness
    return torch.clamp(sigma, min=eps)


# ----------------------------
# Truncated (by discretization) + renormalized Gaussian over bin centers
# ----------------------------
def gaussian_soft_targets(
    labels: torch.Tensor,      # [B]
    bin_centers: torch.Tensor, # [K]
    sigma: torch.Tensor,       # [B] (dynamic) or scalar broadcastable
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns q: [B, K], sums to 1.0 across bins.
    """
    labels = labels.view(-1, 1)          # [B, 1]
    centers = bin_centers.view(1, -1)    # [1, K]
    sigma2 = (sigma.view(-1, 1) ** 2)    # [B, 1]

    w = torch.exp(-0.5 * (labels - centers) ** 2 / (sigma2 + eps))
    w = torch.clamp(w, min=eps)
    w = w / w.sum(dim=1, keepdim=True)
    return w


# ----------------------------
# Freeze/unfreeze callback
# ----------------------------
class UnfreezeBackboneCallback(TrainerCallback):
    """Unfreeze backbone after a specified number of epochs (supports fractional epochs)."""
    
    def __init__(self, model: RexBERTDistributionReranker, unfreeze_after_epoch: float = 0.1):
        self.model_ref = model
        self.unfreeze_after_epoch = unfreeze_after_epoch
        self.did_unfreeze = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.did_unfreeze:
            return control
        if state.epoch is not None and state.epoch >= self.unfreeze_after_epoch:
            for p in self.model_ref.backbone.parameters():
                p.requires_grad = True
            self.did_unfreeze = True
            print(f"[callback] Unfroze backbone at epoch={state.epoch:.4f} (threshold: {self.unfreeze_after_epoch})")
        return control


# ----------------------------
# Custom Trainer: KL + lambda * mean MSE, dynamic sigma
# ----------------------------
class DistributionHybridTrainer(Trainer):
    def __init__(
        self,
        *args,
        bin_centers: torch.Tensor,
        transitions: List[float],
        sigma_min: float,
        sigma_max: float,
        sigma_delta: float,
        lambda_mean: float,
        head_lr: float,
        backbone_lr: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._bin_centers = bin_centers
        self._transitions = transitions
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_delta = sigma_delta
        self._lambda_mean = lambda_mean
        self._head_lr = head_lr
        self._backbone_lr = backbone_lr

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()  # [B]
        outputs = model(**inputs)
        logits = outputs.logits  # [B, K]

        # predicted distribution
        log_probs = F.log_softmax(logits, dim=-1)   # [B, K]
        probs = log_probs.exp()                     # [B, K]

        # dynamic sigma and target distribution
        y = torch.clamp(labels.to(log_probs.device), 0.0, 1.0)
        sigma = sigma_from_transitions(
            y=y,
            transitions=self._transitions,
            sigma_min=self._sigma_min,
            sigma_max=self._sigma_max,
            delta=self._sigma_delta,
        )
        target = gaussian_soft_targets(
            labels=y,
            bin_centers=self._bin_centers.to(log_probs.device),
            sigma=sigma,
        )

        # KL(target || pred)
        kl_loss = F.kl_div(log_probs, target, reduction="batchmean")

        # mean regression auxiliary loss
        bin_centers = self._bin_centers.to(probs.device)
        pred_mean = (probs * bin_centers.view(1, -1)).sum(dim=1)  # [B]
        mean_mse = F.mse_loss(pred_mean, y)

        loss = kl_loss + self._lambda_mean * mean_mse

        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer(self):
        """
        Ensure optimizer includes all params (even frozen), and apply separate LRs:
        - backbone_lr for backbone
        - head_lr for score_head
        """
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
# Metrics: mean + uncertainty proxy + NDCG
# ----------------------------
def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def compute_ndcg_at_k(relevance_scores: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k for a single query.
    relevance_scores: ground truth relevance [N]
    predicted_scores: predicted scores [N]
    k: cutoff
    """
    if len(relevance_scores) == 0:
        return 0.0
    
    # Sort by predicted scores (descending)
    sorted_indices = np.argsort(predicted_scores)[::-1][:k]
    sorted_relevance = relevance_scores[sorted_indices]
    
    # DCG@k
    dcg = np.sum(sorted_relevance / np.log2(np.arange(2, len(sorted_relevance) + 2)))
    
    # Ideal DCG@k (sort by true relevance)
    ideal_indices = np.argsort(relevance_scores)[::-1][:k]
    ideal_relevance = relevance_scores[ideal_indices]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def compute_metrics_builder(num_bins: int, val_queries: Optional[List[str]] = None):
    """
    Build compute_metrics function with NDCG support.
    
    Args:
        num_bins: number of bins for distribution
        val_queries: list of query strings for validation set (in order) for NDCG computation
    """
    bin_centers = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids.astype(np.float32)
        probs = softmax_np(logits, axis=-1)
        pred_mean = (probs * bin_centers[None, :]).sum(axis=1)

        # Basic metrics
        mse = float(np.mean((pred_mean - labels) ** 2))
        mae = float(np.mean(np.abs(pred_mean - labels)))
        pred_var = float(np.mean((probs * (bin_centers[None, :] - pred_mean[:, None]) ** 2).sum(axis=1)))
        entropy = float(np.mean(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum(axis=1)))
        
        metrics = {
            "mse": mse,
            "mae": mae,
            "mean_pred_variance": pred_var,
            "mean_entropy": entropy
        }
        
        # Compute NDCG if query information is available
        if val_queries is not None and len(val_queries) == len(labels):
            # Group by query
            query_to_indices = {}
            for idx, query in enumerate(val_queries):
                if query not in query_to_indices:
                    query_to_indices[query] = []
                query_to_indices[query].append(idx)
            
            # Compute NDCG@5 and NDCG@10 for each query
            ndcg5_scores = []
            ndcg10_scores = []
            
            for query, indices in query_to_indices.items():
                indices = np.array(indices)
                query_labels = labels[indices]
                query_preds = pred_mean[indices]
                
                if len(indices) >= 1:
                    ndcg5 = compute_ndcg_at_k(query_labels, query_preds, k=5)
                    ndcg10 = compute_ndcg_at_k(query_labels, query_preds, k=10)
                    ndcg5_scores.append(ndcg5)
                    ndcg10_scores.append(ndcg10)
            
            if ndcg5_scores:
                metrics["ndcg@5"] = float(np.mean(ndcg5_scores))
                metrics["ndcg@10"] = float(np.mean(ndcg10_scores))
        
        return metrics

    return compute_metrics


def parse_floats_csv(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="thebajajra/RexBERT-base")
    parser.add_argument("--dataset_name", type=str, default="thebajajra/Amazebay-reranker-training-data",
                        help="HuggingFace dataset name with train/validation splits")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)

    parser.add_argument("--num_bins", type=int, default=11)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=["cls", "mean"],
                        help="Pooling strategy: 'cls' uses [CLS] token, 'mean' averages all tokens (recommended for retrieval/reranking)")

    # Dynamic sigma params
    parser.add_argument("--transitions", type=str, default="0.2,0.5,0.8",
                        help="CSV list of transition points in [0,1], e.g. '0.2,0.5,0.8'")
    parser.add_argument("--sigma_min", type=float, default=0.04)
    parser.add_argument("--sigma_max", type=float, default=0.12)
    parser.add_argument("--sigma_delta", type=float, default=0.08)

    # Hybrid loss weight
    parser.add_argument("--lambda_mean", type=float, default=10.0)

    # LRs
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)

    # Backbone unfreezing
    parser.add_argument("--unfreeze_backbone_after", type=float, default=0.1,
                        help="Unfreeze backbone after this many epochs (supports fractional, e.g. 0.1 = 10%% of first epoch)")

    # Training setup
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
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

    # Weights & Biases logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexbert-reranker", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (default: auto-generated)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team name")

    args = parser.parse_args()

    transitions = parse_floats_csv(args.transitions)
    assert all(0.0 <= t <= 1.0 for t in transitions), "All transitions must be in [0,1]."
    assert 0.0 < args.sigma_min <= args.sigma_max, "sigma_min must be >0 and <= sigma_max."
    assert args.sigma_delta > 0.0, "sigma_delta must be > 0."

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup Weights & Biases if requested
    if args.use_wandb:
        import wandb
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        print(f"W&B logging enabled: project={args.wandb_project}, run_name={args.wandb_run_name or 'auto'}")

    # Load tokenizer (standard BERT-style, right padding)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from HuggingFace: expects fields query, title, description, relevance_score
    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)
    
    # Ensure relevance_score column is float32
    ds = ds.cast_column("relevance_score", Value("float32"))
    
    # Store validation queries for NDCG computation (before preprocessing removes them)
    val_queries = None
    if "validation" in ds:
        val_queries = ds["validation"]["query"]
        print(f"Stored {len(val_queries)} validation queries for NDCG computation")

    def preprocess_batch(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        # Format: Query: {query} [SEP] Title: {title} \n Description: {description}
        texts = [
            (
                f"Query: {q}",
                f"Title: {t}\nDescription: {d}"
            )
            for q, t, d in zip(batch["query"], batch["title"], batch["description"])
        ]

        enc = tokenizer(
            [t[0] for t in texts],  # first part: Query: {query}
            [t[1] for t in texts],  # second part: Title: {title} \n Description: {description}
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": batch["relevance_score"],
        }

    ds = ds.map(preprocess_batch, batched=True, remove_columns=["query", "parent_asin", "title", "description", "combined_text", "relevance_label", "relevance_score", "cluster_id"],num_proc=16, batch_size=10000)

    def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return tokenizer.pad(features, padding=True, return_tensors="pt")

    model = RexBERTDistributionReranker(
        backbone_name=args.model_name_or_path,
        num_bins=args.num_bins,
        dropout=args.dropout,
        pooling_strategy=args.pooling_strategy,
        torch_dtype=("bfloat16" if args.bf16 else ("float16" if args.fp16 else None)),
    )

    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    # Freeze backbone for epoch 0 (head-only warm start)
    for p in model.backbone.parameters():
        p.requires_grad = False

    bin_centers = torch.linspace(0.0, 1.0, args.num_bins, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.wandb_run_name if args.use_wandb else None,
        remove_unused_columns=False,
    )

    callbacks = [UnfreezeBackboneCallback(model, unfreeze_after_epoch=args.unfreeze_backbone_after)]

    trainer = DistributionHybridTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collate,
        compute_metrics=compute_metrics_builder(args.num_bins, val_queries=val_queries),
        callbacks=callbacks,
        bin_centers=bin_centers,
        transitions=transitions,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_delta=args.sigma_delta,
        lambda_mean=args.lambda_mean,
        head_lr=args.head_lr,
        backbone_lr=args.backbone_lr,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
