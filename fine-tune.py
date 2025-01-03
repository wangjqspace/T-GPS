import torch
import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, set_seed
from peft import get_peft_model, LoraConfig, TaskType
from Bio import SeqIO
from datasets import Dataset
import random
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import os
import random

os.environ["WANDB_MODE"] = "disabled"

# Step 0: Set random seed for reproducibility
random_seed = 8  # Specify your desired seed value here
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
set_seed(random_seed)  # Ensure reproducibility in Transformers

# Step 1: Load the pre-trained saprot 650M model and tokenizer from Hugging Face
model_name = "/path/to/base-model/SaProt_650M_AF2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Regression task

# Step 2: Apply LoRA adaptation to the Hugging Face saprot 650M model
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,   # Sequence classification or regression task
    inference_mode=False,         # Training mode
    r=4,                         # Low-rank dimension
    lora_alpha=1,                # Scaling parameter
    lora_dropout=0.1,             # Dropout rate for LoRA
    target_modules=["query", "key", "value", "dense"]  # Adjusted target modules for LoRA
)
model = get_peft_model(model, lora_config)

# Step 3: Explicitly freeze non-adapted layers
for name, param in model.named_parameters():
    if not any(target in name for target in ["query", "key", "value", "dense"]):
        param.requires_grad = False

# Ensure the regression head (classifier) is trainable
for param in model.classifier.parameters():
    param.requires_grad = True

# Step 4: Load protein sequences, labels, and set information from the FASTA file
fasta_file = "/path/to/data/mmseq2cluster_noAt_with_stage.fasta"  # Replace with your FASTA file path

def parse_fasta(fasta_file):
    train_sequences = []
    val_sequences = []
    test_sequences = []
    train_labels = []
    val_labels = []
    test_labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        header = record.description
        target_str = header.split("TARGET=")[-1]
        label = float(target_str.split()[0])  # Convert label to float for regression
        set_str = header.split("SET=")[-1].split()[0]
        if set_str == "train":
            train_sequences.append(sequence)
            train_labels.append(label)
        elif set_str == "val":
            val_sequences.append(sequence)
            val_labels.append(label)
        elif set_str == "test":
            test_sequences.append(sequence)
            test_labels.append(label)
    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels

train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = parse_fasta(fasta_file)

# Step 5: Convert sequences and labels into Hugging Face Datasets
def create_dataset(sequences, labels):
    data = {"sequence": sequences, "label": labels}
    return Dataset.from_dict(data)

train_dataset = create_dataset(train_sequences, train_labels)
val_dataset = create_dataset(val_sequences, val_labels)
test_dataset = create_dataset(test_sequences, test_labels)

# Optionally select a random fraction of the train and validation sets
fraction = 1  # Set the fraction to use for training and validation (e.g., 0.5 for 50%)
train_indices = random.sample(range(len(train_dataset)), int(fraction * len(train_dataset)))
val_indices = random.sample(range(len(val_dataset)), int(fraction * len(val_dataset)))
train_dataset = train_dataset.select(train_indices)
val_dataset = val_dataset.select(val_indices)

# Tokenize using Hugging Face tokenizer with proper padding and truncation
max_length = 1024  # Adjustable sequence length
def preprocess_function(examples):
    return tokenizer(
        examples["sequence"], 
        truncation=True, 
        padding="max_length", 
        max_length=max_length
    )

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Step 6: Define training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Save metrics after each epoch
    logging_dir="./logs",
    logging_strategy="epoch",
    per_device_train_batch_size=2,  # Increased batch size
    per_device_eval_batch_size=2,   # Increased batch size for evaluation
    num_train_epochs=20,  # Increased epochs to allow for more training
    save_strategy="epoch",  # Save model at the end of each epoch
    save_steps=1000,
    save_total_limit=2,
    gradient_accumulation_steps=4,  # Gradient accumulation to simulate larger batch size
    fp16=True,  # Mixed precision training
    dataloader_num_workers=4,  # Adjust number of workers based on CPU capacity
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # Use evaluation loss as the metric for best model
    greater_is_better=False,  # Lower eval_loss is better
)

# Early stopping callback (patience defines how many evaluations to wait before stopping)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  # Stop after 3 evaluations without improvement

# Clear CUDA cache before starting training to free up memory
torch.cuda.empty_cache()

# Step 7: Update the AdamW optimizer with new parameters
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,                 # Learning rate
    betas=(0.9, 0.98),      # Beta values
    eps=1e-8,                # Epsilon
    weight_decay=0.01        # Weight decay
)

# Custom callback to monitor training and log metrics after each epoch
class MetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        with open("./results/metrics.json", "a") as f:
            json.dump(metrics, f)
            f.write("\n")

# Define the compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()  # Squeeze in case there's an extra dimension
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    spearman_corr, _ = spearmanr(labels, predictions)
    pearson_corr, _ = pearsonr(labels, predictions)
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Spearman Correlation": float(spearman_corr),
        "Pearson Correlation": float(pearson_corr)
    }

# Initialize Trainer with tokenized dataset, modified model, and custom optimizer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,  # Use validation set for evaluation during training
    optimizers=(optimizer, None),  # Custom optimizer, no scheduler
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback, early_stopping_callback]  # Add the custom callback for logging metrics and early stopping
)

# Step 8: Fine-tune the model
trainer.train()

# Step 9: Save the fine-tuned model and tokenizer
save_dir = "./results"  # Directory to save the fine-tuned model

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Save the model
model.save_pretrained(save_dir)

# Save the tokenizer
tokenizer.save_pretrained(save_dir)

print(f"Fine-tuned model saved to {save_dir}")

# Step 10: Save the inference results for the validation and test sets as separate FASTA files
def save_predictions_to_fasta(dataset, predictions, output_file):
    with open(output_file, "w") as f:
        for seq, pred, target in zip(dataset["sequence"], predictions, dataset["label"]):
            header = f">TARGET={target:.4f} PREDICT={pred:.4f}"  # Include original TARGET and predicted value as PREDICT
            f.write(f"{header}\n{seq}\n")

# Inference on validation set
val_predictions, _, _ = trainer.predict(tokenized_val_dataset)
val_predictions = val_predictions.squeeze()
save_predictions_to_fasta(val_dataset, val_predictions, "./results/val_predictions.fasta")

# Inference on test set
test_predictions, _, _ = trainer.predict(tokenized_test_dataset)
test_predictions = test_predictions.squeeze()
save_predictions_to_fasta(test_dataset, test_predictions, "./results/test_predictions.fasta")

print("Inference results saved successfully!")

