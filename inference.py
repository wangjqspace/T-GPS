import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from Bio import SeqIO
import re

# Paths to the base model, LoRA adapter, and input/output FASTA files
model_name_dir = "/path/to/base-model/SaProt_650M_AF2"  # Replace with pre-trained base model directory
model = AutoModelForSequenceClassification.from_pretrained(model_name_dir, num_labels=1)  # Regression task
adapter_dir = "./results"  # Replace with the directory containing LoRA adapter files
input_fasta = "/path/to/new-species-fasta/saprot_At.fasta"  # Input FASTA file
output_fasta = "./saprot_At_predictions.fasta"  # Output FASTA file with predictions
batch_size = 64  # Define batch size for processing

# Step 1: Load the tokenizer from the base model directory
tokenizer = AutoTokenizer.from_pretrained(model_name_dir)

# Step 2: Load the pre-trained base model
print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_name_dir, num_labels=1)
print("Base model loaded successfully.")

# Step 3: Load the LoRA adapter
print("Loading LoRA adapter...")
config = PeftConfig.from_pretrained(adapter_dir)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()  # Set the model to evaluation mode
print("LoRA adapter loaded successfully.")

# Step 4: Define the batched inference function
def run_inference_batched(model, tokenizer, sequences, batch_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Run inference using the model and tokenizer in batches.

    Args:
        model: The model with the LoRA adapter loaded.
        tokenizer: The tokenizer.
        sequences: List of input sequences for inference.
        batch_size: Number of sequences to process in a single batch.
        device: Device to run the inference on ("cuda" or "cpu").
    
    Returns:
        List of predictions.
    """
    model.to(device)
    predictions = []
    total_sequences = len(sequences)
    for i in range(0, total_sequences, batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_predictions = outputs.logits.squeeze(-1).tolist()  # Adjust for regression tasks
            predictions.extend(batch_predictions)
        
        # Print progress
        completed = min(i + batch_size, total_sequences)
        percentage = (completed / total_sequences) * 100
        print(f"Progress: {completed}/{total_sequences} sequences ({percentage:.2f}%) completed.", end="\r")
    print()  # Ensure the final progress update appears on a new line
    return predictions

# Step 5: Read sequences and headers from the input FASTA file
sequences = []
headers = []
for record in SeqIO.parse(input_fasta, "fasta"):
    sequences.append(str(record.seq))
    headers.append(record.description)  # Keep full description for existing TARGET=value

# Step 6: Run inference on the sequences in batches
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running inference in batches of {batch_size}...")
predictions = run_inference_batched(model, tokenizer, sequences, batch_size, device=device)

# Step 7: Write the output to a new FASTA file
print(f"Saving predictions to {output_fasta}...")
with open(output_fasta, "w") as output_file:
    for header, seq, pred in zip(headers, sequences, predictions):
        # Append the PREDICT=value to the header
        updated_header = f"{header} PREDICT={pred:.6f}"
        output_file.write(f">{updated_header}\n{seq}\n")

print(f"Predictions saved to {output_fasta}")

