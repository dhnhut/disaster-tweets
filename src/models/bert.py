from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import Dataset
from sklearn.metrics import classification_report

from src import setup

# https://huggingface.co/google-bert/bert-base-uncased
# bert-base-uncased - 110M


def create_datasets(df_train, df_val, df_test):
    # Convert to Hugging Face Datasets
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    test_ds = Dataset.from_pandas(df_test)

    return train_ds, val_ds, test_ds


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["tweet_text"],
        padding="max_length",
        truncation=True,  # tweet_text is normally short
        max_length=64,
    )


def max_length_dist(
    df, field, tokenizer, frac: float = 0.1, random_state: int = setup.RANDOM_SEED
):
    ds = Dataset.from_pandas(df.sample(frac=frac, random_state=random_state))

    token_lengths = [len(tokenizer.tokenize(str(text))) for text in ds[field]]

    df_lengths = pd.DataFrame({"length": token_lengths})

    print("90th percentile:", df_lengths["length"].quantile(0.90))
    print("95th percentile:", df_lengths["length"].quantile(0.95))
    print("99th percentile:", df_lengths["length"].quantile(0.99))
    print("Absolute Maximum length:", df_lengths["length"].max())


def format_dataset(dataset):
    # The HF Trainer API expects strictly the named `labels`
    # Rename target column to 'labels'
    dataset = dataset.rename_column("informative", "labels")

    # Cast boolean values to integers (True -> 1, False -> 0)
    dataset = dataset.map(lambda x: {"labels": int(x["labels"])})

    # Set PyTorch tensor format for the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def load_or_tokenize(
    ds_train, ds_val, ds_test, tokenizer, save_path: Path, force_retokenize=False
):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train_tokenized"
    val_path = save_path / "val_tokenized"
    test_path = save_path / "test_tokenized"

    if (
        (train_path).exists()
        and (val_path).exists()
        and (test_path).exists()
        and not force_retokenize
    ):
        print("Loading tokenized datasets from disk...")

        train_tokenized = Dataset.load_from_disk(train_path)
        val_tokenized = Dataset.load_from_disk(val_path)
        test_tokenized = Dataset.load_from_disk(test_path)

    else:
        print("Tokenizing datasets...")
        # Apply in batches for efficiency
        train_tokenized = format_dataset(
            ds_train.map(lambda x: tokenize_function(tokenizer, x), batched=True)
        )
        val_tokenized = format_dataset(
            ds_val.map(lambda x: tokenize_function(tokenizer, x), batched=True)
        )
        test_tokenized = format_dataset(
            ds_test.map(lambda x: tokenize_function(tokenizer, x), batched=True)
        )

        # Save individual datasets to a specified directory
        print(f"Saving tokenized datasets to {save_path}...")
        train_tokenized.save_to_disk(train_path)
        val_tokenized.save_to_disk(val_path)
        test_tokenized.save_to_disk(test_path)

    return train_tokenized, val_tokenized, test_tokenized


def finetune(train_tokenized, val_tokenized, configs: dict):
    batch_size = configs["batch_size"]
    model = configs["bert"]
    device = configs["device"]
    optimizer = configs["optimizer"]
    num_epochs = configs["num_epochs"]
    patience = configs["patience"]

    train_loader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(val_tokenized, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    print(f"Starting {model.__class__.__name__} fine-tuning...")
    print(f"Using device: {device}")
    print(f"Number of training samples: {len(train_tokenized)}")
    print(f"Number of validation samples: {len(val_tokenized)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Early stopping patience: {patience} epochs")
    print("-" * 50)

    model.to(device)
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = batch["labels"].to(device).view(-1).long()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation at end of each epoch
        model.eval()
        correct = 0
        total = 0
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(
                eval_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"
            ):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = batch["labels"].to(device).view(-1).long()

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                eval_losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = sum(eval_losses) / len(eval_losses)
        accuracy = correct / total
        val_loss_history.append(avg_loss)
        val_acc_history.append(accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_loss:.4f} | "
            f"Val Acc: {accuracy:.4f}"
        )

        # Early stopping + keep best model weights
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Restore best model before test prediction
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Loaded best model with Val Loss: {best_val_loss:.4f}")

    return model, train_loss_history, val_loss_history, val_acc_history


def predict(model, test_tokenized, device):
    test_loader = DataLoader(test_tokenized, batch_size=16)
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting on Test Set"):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds


def report_metrics(test_tokenized, predictions, labels="labels"):
    y_true = np.array(test_tokenized[labels])
    y_pred = np.array(predictions)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
