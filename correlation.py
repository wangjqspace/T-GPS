import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import argparse

# Function to parse FASTA file and extract TARGET and PREDICT values
def parse_fasta(file_path):
    target_values = []
    predict_values = []

    with open(file_path, 'r') as fasta_file:
        for line in fasta_file:
            if line.startswith('>'):
                # Extract TARGET and PREDICT values from the header
                header = line.strip()
                target = float(header.split('TARGET=')[1].split()[0])
                predict = float(header.split('PREDICT=')[1].split()[0])
                target_values.append(target)
                predict_values.append(predict)

    return target_values, predict_values

# Function to calculate and plot correlations
def plot_correlations(target_values, predict_values, output_file="correlation_saprot_At_predictions.txt"):
    # Calculate Pearson correlation coefficient
    pcc, _ = pearsonr(target_values, predict_values)

    # Calculate Spearman rank correlation coefficient
    spearman_rho, _ = spearmanr(target_values, predict_values)

    # Save results to file
    with open(output_file, 'w') as result_file:
        result_file.write(f"Pearson Correlation Coefficient (PCC): {pcc:.3f}\n")
        result_file.write(f"Spearman Rank Correlation Coefficient (Rho): {spearman_rho:.3f}\n")

    # Plot the values
    plt.figure(figsize=(8, 6))
    plt.scatter(target_values, predict_values, alpha=0.7)
    plt.title(f"PCC: {pcc:.3f}, Spearman Rho: {spearman_rho:.3f}", fontsize=14)
    plt.xlabel("TARGET", fontsize=12)
    plt.ylabel("PREDICT", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    print(f"Pearson Correlation Coefficient (PCC): {pcc:.3f}")
    print(f"Spearman Rank Correlation Coefficient (Rho): {spearman_rho:.3f}")

# Main function to handle command-line input
def main():
    parser = argparse.ArgumentParser(description="Parse a FASTA file and calculate correlations between TARGET and PREDICT values.")
    parser.add_argument("--input_fasta", required=True, help="Path to the input FASTA file.")

    args = parser.parse_args()

    # Parse the FASTA file
    targets, predicts = parse_fasta(args.input_fasta)

    # Plot correlations and save results
    plot_correlations(targets, predicts)

if __name__ == "__main__":
    main()
