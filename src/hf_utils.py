import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from datasets import Dataset
from sklearn.metrics import classification_report

from . import setup


def tokenize_function(tokenizer, examples):
    return tokenizer(
        examples["tweet_text"],
        padding="max_length",
        truncation=True,  # tweet_text is normally short
        max_length=64,
    )


def create_datasets(df_train, df_val, df_test):
    # Convert to Hugging Face Datasets
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    test_ds = Dataset.from_pandas(df_test)

    return train_ds, val_ds, test_ds


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


def load_or_tokenize(
    ds_train,
    ds_val,
    ds_test,
    tokenizer,
    save_path: Path,
    force_retokenize=False,
    format_dataset=None,
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
        train_tokenized = ds_train.map(
            lambda x: tokenize_function(tokenizer, x), batched=True
        )
        val_tokenized = ds_val.map(
            lambda x: tokenize_function(tokenizer, x), batched=True
        )
        test_tokenized = ds_test.map(
            lambda x: tokenize_function(tokenizer, x), batched=True
        )

        if format_dataset is not None:
            train_tokenized = format_dataset(train_tokenized)
            val_tokenized = format_dataset(val_tokenized)
            test_tokenized = format_dataset(test_tokenized)

        # Save individual datasets to a specified directory
        print(f"Saving tokenized datasets to {save_path}...")
        train_tokenized.save_to_disk(train_path)
        val_tokenized.save_to_disk(val_path)
        test_tokenized.save_to_disk(test_path)

    return train_tokenized, val_tokenized, test_tokenized


def plot_fine_tune_history(train_loss_history, val_loss_history, val_f1_history, val_recall_history, val_precision_history):
    if len(train_loss_history) == 0 or len(val_loss_history) == 0:
        print("No training history found. Run the fine-tuning cell first.")
        return

    epochs = range(1, len(train_loss_history) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    axes[0].plot(epochs, train_loss_history, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss_history, marker="o", label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs, val_f1_history, marker="o", label="Val F1 Score")
    axes[1].plot(epochs, val_recall_history, marker="o", label="Val Recall")
    axes[1].plot(epochs, val_precision_history, marker="o", label="Val Precision")
    axes[1].set_title("Validation F1, Recall and Precision Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    
    # Dynamic Y-Axis Calculation
    min_score = min(min(val_f1_history), min(val_recall_history), min(val_precision_history))
    max_score = max(max(val_f1_history), max(val_recall_history), max(val_precision_history))

    # 5% padding to give the data points some breathing room
    padding = 0.05 
    y_min = max(0.0, min_score - padding)
    y_max = min(1.0, max_score + padding)
    
    axes[1].set_ylim(y_min, y_max)
    
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def group_report_metrics(
    df, predictions, labels="informative", group_by="disaster_type"
):
    df["prediction"] = predictions
    for subset_name, subset_data in df.groupby(group_by):
        print(f"{'='*50}")
        print(f" Classification Report for Subset: {subset_name}")
        print(f"{'-'*50}")

        y_true_subset = subset_data[labels]
        y_pred_subset = subset_data["prediction"]
        report = classification_report(y_true_subset, y_pred_subset, digits=4)
        print(report)
