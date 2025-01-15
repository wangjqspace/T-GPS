# 🧬 T-GPS fine-tune model to predict pan-plant protein thermal stability
A T-GPS fine-tuned Saprot model for plant protein thermal stability prediction.

## 🌟 Features
- Predicts thermal stability based on the structure-aware tokens from Saprot model.
- Supports batch inference.
- Support inference based on a single model or an ensemble model

## Notes
Example data for model fine-tuning are provided in the folder "Data/train"
Example data for predicting the thermostability are provided in the folder "Data/predict" 
The T-GPS fine-tune SaProt models (adapters) are in the folder "adapters".

## ⚙️ Installation
#Download base model from Huggingface
https://huggingface.co/SaProtHub/Model-Thermostability-650M/tree/main

```bash
git clone https://github.com/wangjqspace/T-GPS.git
cd T-GPS

#Create a virtual environment
conda create -n tgps python=3.10
conda activate tgps

#Install packages
bash environment.sh
```

## 🧪 Usage
```bash
# Fine-tuning
python fine-tune.py

# Predict thermal stability using a single model
python inference.py --input_fasta input.fasta --output_fasta output.fasta

# Predict thermal stability using an ensemble model
python inference_ensemble.py --input_fasta input.fasta --output_fasta output.fasta

# Calculate the correlation between predicted and experimental result.
python correlation.py
```
