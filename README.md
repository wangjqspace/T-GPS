# 🧬 Protein Prediction Model
A T-GPS fine-tuned Saprot model for plant protein thermal stability prediction.

## 🌟 Features
- Predicts thermal stability based on the structure-aware tokens from Saprot model.
- Supports batch inference

## ⚙️ Installation

# T-GPS
#Download base model from Huggingface
https://huggingface.co/SaProtHub/Model-Thermostability-650M/tree/main
```bash
git clone https://github.com/wangjqspace/T-GPS.git
cd T-GPS
#Create a virtual environment

conda create -n tgps python=3.10
conda activate tgps
bash environment.sh  

#to fine tune the SaProt model
	python fine-tune.py

#to predict thermostability of proteins 
	python inference.py --input_fasta input_fasta --output_fasta output.fasta

#to predict thermostability of proteins based on ensemble models
	python inference_ensemble.py --input_fasta input_fasta --output_fasta output.fasta

#To calculate the correlation between predicted and experimental result.
	python correlation.py

The data for model fine-tuning and evaluation are in the folder "Data/train"
The data for calculating the thermostability of the proteins of the hold-out specie are in the folder "Data/new-species" 
The T-GPS fine-tune SaProt models (adapters) is in the folder "adapters".
