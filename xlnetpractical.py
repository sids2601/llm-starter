import random

import pandas as pd
import numpy as np
import re
from cleantext import clean
from tensorflow.python.ops.nn_ops import top_k
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer, pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datasets
import evaluate


def tokenize_function(examples):
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    data_train = pd.read_csv("emotion-labels-train.csv")
    data_test = pd.read_csv("emotion-labels-test.csv")
    data_val = pd.read_csv("emotion-labels-val.csv")

    data = pd.concat([data_train, data_test, data_val], ignore_index=True)
    data['text_clean'] = data['text'].apply(lambda x: clean(x, no_emoji=True))
    data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r"[^\w\s]", '', x))
    data_label = data.groupby('label')
    data = pd.DataFrame(data_label.apply(lambda x: x.sample(data_label.size().min()))).reset_index(drop=True)
    data['label_int'] = LabelEncoder().fit_transform(data['label'])
    NUM_LABELS = 4
    train_split, test_split = train_test_split(data, train_size=0.8)
    train_split, val_split = train_test_split(train_split, train_size=0.9)
    train_df = pd.DataFrame({
        "label": train_split.label_int.values,
        "text": train_split.text_clean.values
    })
    test_df = pd.DataFrame({
        "label": test_split.label_int.values,
        "text": test_split.text_clean.values
    })
    train_df = datasets.Dataset.from_dict(train_df)
    test_df = datasets.Dataset.from_dict(test_df)
    dataset_dict = datasets.DatasetDict({"train": train_df, "test": test_df})
    tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=NUM_LABELS,
                                                           id2label={0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'})
    metric = evaluate.load("accuracy")
    training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=3)
    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(100))
    small_test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(100))
    trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset,
                      eval_dataset=small_test_dataset, compute_metrics=compute_metrics)
    trainer.train()
    trainer.evaluate()
    model.save_pretrained('fine_tuned_model')
    fine_tuned_model = XLNetForSequenceClassification.from_pretrained('fine_tuned_model')
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    clf = pipeline('text-classification', model=fine_tuned_model, tokenizer=tokenizer)
    val_split = val_split.reset_index(drop=True)
    print(val_split['text_clean'][0])
    print(clf(val_split['text_clean'][0]))
