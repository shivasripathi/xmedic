# src/nlp/train_nlp.py
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset # Hugging Face's dataset library

# --- 1. Load Pre-trained Model and Tokenizer ---
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1-squad"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# --- 2. Prepare Your Dataset ---
# In a real scenario, you'd load your contexts and create question/answer pairs.
# For this example, we'll create a tiny dummy dataset.
dummy_data = {
    'question': ["What is the incubation period of COVID-19?", "What are the symptoms of melanoma?"],
    'context': [
        "The incubation period for COVID-19, which is the time between exposure to the virus and symptom onset, is on average 5-6 days, but can be as long as 14 days.",
        "The most common sign of melanoma is the appearance of a new mole or a change in an existing mole. This can be remembered by ABCDE: Asymmetry, Border, Color, Diameter, and Evolving."
    ],
    'answers': [
        {'text': ['5-6 days'], 'answer_start': [100]},
        {'text': ['the appearance of a new mole'], 'answer_start': [35]}
    ]
}

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict(dummy_data)

# --- 3. Tokenize the Dataset ---
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
    # Further processing to map answer start/end to token positions...
    # This is a complex step, simplified here.
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)

# --- 4. Fine-Tune the Model ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_train # Use same for simplicity
)

# Start training!
trainer.train()

# --- 5. Save the Fine-Tuned Model ---
MODEL_SAVE_PATH = '../models/biobert_qa_finetuned'
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Fine-tuned NLP model saved to {MODEL_SAVE_PATH}")