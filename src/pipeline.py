from pathlib import Path
import csv
import math
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import evaluate

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel

import configuration

from src import hf_utils, setup
from src.models import bert, llama_bi


def stage_1_bert_binary(df, model_name, model_path, tokenized_path, device):
    print("Stage 1: Binary Classification with BERT")
    model = BertForSequenceClassification.from_pretrained(model_path)
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    model.to(device)

    ds = Dataset.from_pandas(df)

    tokenized = hf_utils.tokenize(
        ds, bert_tokenizer, tokenized_path, bert.format_dataset
    )

    predictions, confidences = bert.predict(
        model, tokenized, device, confidence_scores=True
    )

    hf_utils.report_metrics(tokenized, predictions)

    return predictions, confidences


def stage_2_llama_binary(df, model_name, model_path, tokenized_path, device):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float16,  # or torch.bfloat16 depending on your hardware
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Llama 3.2 does not have a native padding token, which is required for sequence classification batching.
    # Use the EOS token for padding and propagate it to model configs to avoid batch-size errors.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base, model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    ds = Dataset.from_pandas(df)

    tokenized = hf_utils.tokenize(
        ds, tokenizer, tokenized_path, llama_bi.format_dataset
    )
    # accuracy_metric = evaluate.load("accuracy")
    # precision_metric = evaluate.load("precision")
    # recall_metric = evaluate.load("recall")
    # f1_metric = evaluate.load("f1")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(tokenized, batch_size=8, collate_fn=data_collator)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            outputs = model(**batch)
            all_logits.append(outputs.logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=-1).numpy()
    predictions = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)
    # test_acc = accuracy_metric.compute(
    #     predictions=predictions, references=tokenized["labels"]
    # )
    # test_precision = precision_metric.compute(
    #     predictions=predictions, references=tokenized["labels"]
    # )
    # test_recall = recall_metric.compute(
    #     predictions=predictions, references=tokenized["labels"]
    # )
    # test_f1 = f1_metric.compute(
    #     predictions=predictions, references=tokenized["labels"]
    # )

    hf_utils.report_metrics(tokenized, predictions)

    return predictions, confidences
