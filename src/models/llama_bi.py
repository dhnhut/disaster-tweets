from datasets import Value


def format_dataset(dataset):
    # The HF Trainer API expects a column named 'labels'.
    dataset = dataset.rename_column("informative", "labels")
    dataset = dataset.map(lambda x: {"labels": int(x["labels"])})
    # Ensure labels are int64 for cross-entropy loss.
    dataset = dataset.cast_column("labels", Value("int64"))
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
