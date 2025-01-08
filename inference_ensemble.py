
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run ensemble inference for regression task on input FASTA file.")
parser.add_argument("--input_fasta", type=str, required=True, help="Path to the input FASTA file.")
parser.add_argument("--output_fasta", type=str, required=True, help="Path to the output FASTA file.")
args = parser.parse_args()

# Paths to the base model and LoRA adapters
model_name_dir = "/home/dc/models/SaProt_650M_AF2"
adapter_dirs = [
    "./results/at_1",
    "./results/os_1",
    "./results/sl_1",
    "./results/zm_1",
    "./results/ch_1"
]
batch_size = 64  # Define batch size for processing

# Step 1: Load the tokenizer from the base model directory
tokenizer = AutoTokenizer.from_pretrained(model_name_dir)

# Step 2: Load the base model and adapters
print("Loading base model and adapters...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_name_dir, num_labels=1)
adapted_models = []
for adapter_dir in adapter_dirs:
    config = PeftConfig.from_pretrained(adapter_dir)
    adapted_model = PeftModel.from_pretrained(base_model, adapter_dir)
    adapted_model.eval()  # Set the model to evaluation mode
    adapted_models.append(adapted_model)
print("Adapters loaded successfully.")

# Step 3: Define the batched inference function
def run_inference_batched(models, tokenizer, sequences, batch_size, device="cuda" if torch.cuda.is_available() else "cpu"):
    for model in models:
        model.to(device)

    averaged_predictions = []
    total_sequences = len(sequences)

    for i in range(0, total_sequences, batch_size):
        batch_sequences = sequences[i:i + batch_size]
        inputs = tokenizer(batch_sequences, padding=True, truncation=True, max_length=1024, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        batch_predictions = []
        with torch.no_grad():
            for model in models:
                outputs = model(**inputs)
                predictions = outputs.logits.squeeze(-1).tolist()  # Extract logits for regression
                batch_predictions.append(predictions)

        # Average predictions across all models
        batch_avg_predictions = np.mean(batch_predictions, axis=0).tolist()
        averaged_predictions.extend(batch_avg_predictions)

        # Print progress
        completed = min(i + batch_size, total_sequences)
        percentage = (completed / total_sequences) * 100
        print(f"Progress: {completed}/{total_sequences} sequences ({percentage:.2f}%) completed.", end="\r")

    print()  # Newline for clean output
    return averaged_predictions

# Step 4: Read sequences and headers from the input FASTA file
sequences = []
headers = []
for record in SeqIO.parse(args.input_fasta, "fasta"):
    sequences.append(str(record.seq))
    headers.append(record.description)  # Keep full description without modifying it

# Step 5: Run ensemble inference on the sequences in batches
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running ensemble inference in batches of {batch_size}...")
averaged_predictions = run_inference_batched(adapted_models, tokenizer, sequences, batch_size, device=device)

# Step 6: Write predictions to the output FASTA file without splitting sequences
print("Saving predictions to output FASTA file...")
with open(args.output_fasta, "w") as output_handle:
    for header, seq, pred in zip(headers, sequences, averaged_predictions):
        new_header = f"{header} PREDICT={pred:.4f}"
        output_handle.write(f">{new_header}\n{seq}\n")

print(f"Predictions saved to {args.output_fasta}")



