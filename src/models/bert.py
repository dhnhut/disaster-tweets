from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
)

from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

from .. import data_utils

# https://huggingface.co/google-bert/bert-base-uncased
# bert-base-uncased - 110M


def model_path(weather_ratio=None, out_topic_ratio=None):
    return f"../models/BERT/{data_utils.get_experiment_ratios_path(weather_ratio, out_topic_ratio)}"


def format_dataset(dataset):
    # The HF Trainer API expects strictly the named `labels`
    # Rename target column to 'labels'
    dataset = dataset.rename_column("informative", "labels")

    # Cast boolean values to integers (True -> 1, False -> 0)
    dataset = dataset.map(lambda x: {"labels": int(x["labels"])})

    # Set PyTorch tensor format for the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def detect_imbalance_strategy(train_labels):
    counts = Counter(train_labels)
    majority_count = max(counts.values())
    minority_count = min(counts.values())

    ir = majority_count / minority_count
    # num_classes = len(counts)

    # Identify which class ID is the minority
    minority_class_id = min(counts, key=counts.get)

    configs = {
        "strategy": "standard",
        "minority_class_id": minority_class_id,
        "class_weights": None,
        "use_sampler": False,
        "use_focal_loss": False,
        "imbalance_ratio": ir,
    }

    print(f"Dataset Imbalance Ratio (IR): {ir:.2f}")

    if ir <= 1.5:
        print("Status: Balanced. Using standard CrossEntropyLoss.")
        return configs

    elif 1.5 < ir <= 5.0:
        print("Status: Moderate Imbalance. Using Class Weights.")
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        configs["strategy"] = "class_weights"
        configs["class_weights"] = torch.tensor(weights, dtype=torch.float)
        return configs

    elif 5.0 < ir < 15:
        print("Status: High Imbalance. Using WeightedRandomSampler.")
        # Calculate sample weights as shown in previous responses
        class_weights_dict = {cls: 1.0 / count for cls, count in counts.items()}
        sample_weights = [class_weights_dict[label] for label in train_labels]

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        configs["strategy"] = "sampler"
        configs["use_sampler"] = True
        configs["train_sampler"] = sampler
        return configs

    else:
        print("Status: Extreme Imbalance. Using Focal Loss.")
        configs["strategy"] = "focal_loss"
        configs["use_focal_loss"] = True

        # Focal loss also benefits from alpha weighting
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(train_labels), y=train_labels
        )
        configs["class_weights"] = torch.tensor(weights, dtype=torch.float)
        return configs


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def finetune(train_tokenized, val_tokenized, configs: dict):
    batch_size = configs["batch_size"]
    model = configs["bert"]
    device = configs["device"]
    optimizer = configs["optimizer"]
    num_epochs = configs["num_epochs"]
    patience = configs["patience"]
    save_path = configs["save_path"]

    # 1. Extract strategy configurations
    strategy = configs.get("strategy", "standard")
    use_sampler = configs.get("use_sampler", False)
    train_sampler = configs.get("train_sampler")
    class_weights = configs.get("class_weights")
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # 2. Configure DataLoader
    train_loader = DataLoader(
        train_tokenized,
        batch_size=batch_size,
        sampler=train_sampler if use_sampler else None,
        shuffle=False if use_sampler else True,
    )
    eval_loader = DataLoader(val_tokenized, batch_size=batch_size)

    # 3. Configure Loss Function
    if strategy == "focal_loss":
        loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
    elif strategy == "class_weights":
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 2. Initialize tracking variables for Recall
    best_val_loss = float("inf")
    best_val_recall = 0.0  # Tracks highest recall
    best_state_dict = None
    epochs_without_improvement = 0

    train_loss_history = []
    val_loss_history = []
    val_f1_history = []
    val_recall_history = []
    val_precision_history = []

    print(f"Starting {model.__class__.__name__} fine-tuning...")
    print(f"Active Strategy: {strategy.upper()}")
    print("-" * 50)

    model.to(device)
    unk_token_id = getattr(model.config, "unk_token_id", 100)

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).view(-1).long()

            # 4. Apply Token Dropout if using Sampler
            if strategy == "sampler":
                prob_matrix = torch.rand(input_ids.shape, device=device)
                drop_mask = (prob_matrix < 0.05) & (input_ids != 0)
                input_ids[drop_mask] = unk_token_id

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        eval_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                eval_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"
            ):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = batch["labels"].to(device).view(-1).long()

                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)
                logits = outputs.logits

                eval_losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = sum(eval_losses) / len(eval_losses)
        val_loss_history.append(avg_loss)

        # 3. Calculate Macro F1 and Recall
        epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        epoch_recall = recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        epoch_precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

        val_f1_history.append(epoch_f1)
        val_recall_history.append(epoch_recall)
        val_precision_history.append(epoch_precision)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_loss:.4f} | "
            f"Val Macro F1: {epoch_f1:.4f} | "
            f"Val Recall: {epoch_recall:.4f} | "
            f"Val Precision: {epoch_precision:.4f}"
        )

        # 4. Early stopping based on Validation Recall
        if epoch_recall > best_val_recall:
            best_val_recall = epoch_recall
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val Recall: {best_val_recall:.4f}"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Loaded best model with Val Recall: {best_val_recall:.4f}")

    if save_path is not None:
        model.save_pretrained(save_path)

    # 5. Return the newly tracked recall history as well
    return (
        model,
        train_loss_history,
        val_loss_history,
        val_f1_history,
        val_recall_history,
        val_precision_history,
    )


def predict(model, test_tokenized, device, confidence_scores=False):
    test_loader = DataLoader(test_tokenized, batch_size=16)
    model.eval()
    all_preds = []
    all_confidences = []
    # all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting:"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            if confidence_scores:
                all_confidences.extend(confs.cpu().numpy())
                # all_probs.extend(probs.cpu().numpy())

    if confidence_scores:
        # return all_preds, all_confidences, all_probs
        return all_preds, all_confidences
    return all_preds


def report_metrics(test_tokenized, predictions, labels="labels"):
    y_true = np.array(test_tokenized[labels])
    y_pred = np.array(predictions)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
