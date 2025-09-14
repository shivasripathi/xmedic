# src/nlp/train_dermatology_qa.py
"""
This script fine-tunes a BioBERT model for Question Answering on a specialized
Dermatology dataset from Kaggle. It is the Python script version of the
1_Dermatology_QA_Model_Training.ipynb notebook.
"""

import os
import pandas as pd
import subprocess
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

# --- Configuration ---
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1-squad"
DATASET_NAME = "muhammadareebkhan/skin-disease-medical-text-data-for-fine-tuning"
DATA_DIR = "../../data/dermatology_qa"
CSV_FILE_NAME = "Skin Disease Medical Text Data.csv"
MODEL_SAVE_PATH = "../../models/dermatology_qa_model"

def download_and_unzip_dataset():
    """
    Downloads the dermatology QA dataset from Kaggle using the Kaggle API.
    """
    print("--- Step 1: Downloading Dataset from Kaggle ---")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if data already exists to avoid re-downloading
    if os.path.exists(os.path.join(DATA_DIR, CSV_FILE_NAME)):
        print("Dataset already found. Skipping download.")
        return

    try:
        command = [
            "kaggle", "datasets", "download",
            "-d", DATASET_NAME,
            "-p", DATA_DIR,
            "--unzip"
        ]
        subprocess.run(command, check=True)
        print("Dataset downloaded and unzipped successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: Could not download dataset from Kaggle.")
        print("Please ensure:")
        print("1. You have the 'kaggle' package installed (`pip install kaggle`).")
        print("2. Your Kaggle API token (`kaggle.json`) is correctly set up.")
        exit()


def load_and_preprocess_data():
    """
    Loads the CSV, renames columns, formats it for QA tasks, and splits it.
    """
    print("\n--- Step 2: Loading and Preprocessing Data ---")
    df = pd.read_csv(os.path.join(DATA_DIR, CSV_FILE_NAME))
    df.rename(columns={'Q': 'question', 'A': 'context'}, inplace=True)

    # For QA, the 'answer' is a snippet from the context.
    # We will treat the entire context as the answer since that's how the dataset is structured.
    df['answers'] = df['context'].apply(lambda x: {'text': [str(x)], 'answer_start': [0]})
    df = df.dropna(subset=['question', 'context']) # Ensure no null values

    print("Original Data Sample:")
    print(df.head())

    full_dataset = Dataset.from_pandas(df)

    # Split into training and testing sets
    train_test_split = full_dataset.train_test_split(test_size=0.1)
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })
    print("\nProcessed Dataset Structure:")
    print(dataset_dict)
    return dataset_dict


def tokenize_dataset(dataset_dict, tokenizer):
    """
    Tokenizes the text data using the provided tokenizer.
    """
    print("\n--- Step 3: Tokenizing Data ---")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
                end_positions.append(idx + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, remove_columns=dataset_dict['train'].column_names)
    print("Tokenization complete.")
    return tokenized_datasets


def train_model(tokenized_datasets, tokenizer):
    """
    Sets up the Trainer and fine-tunes the model.
    """
    print("\n--- Step 4: Fine-Tuning Model ---")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir="../../models/dermatology_qa_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting model fine-tuning...")
    trainer.train()
    print("Fine-tuning complete.")
    
    print("\n--- Step 5: Saving Model ---")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Dermatology QA model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    # Execute the full pipeline
    download_and_unzip_dataset()
    
    dataset = load_and_preprocess_data()
    
    # Initialize the tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_data = tokenize_dataset(dataset, tokenizer)
    
    train_model(tokenized_data, tokenizer)
    
    print("\n--- Training Pipeline Finished Successfully ---")
