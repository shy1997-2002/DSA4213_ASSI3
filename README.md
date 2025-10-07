# IMDB Sentiment Classification with Parameter-Efficient Fine-Tuning (PEFT)  
  
This repository contains the code and configurations used to compare **Full Fine-tuning**, **LoRA**, and **BitFit** methods on the **IMDB sentiment analysis dataset** using the DistilBERT model.  
The goal of this project is to analyze the trade-offs between different fine-tuning strategies in terms of **performance**, **parameter efficiency**, and **computational cost**.
---  
## Project Overview  
  
This project was developed for the **DSA4213: Advanced Topics in AI Systems** course assignment.    
It demonstrates the implementation and evaluation of several **Parameter-Efficient Fine-Tuning (PEFT)** methods using Hugging Face’s `transformers` and `peft` libraries.
---  
## Install dependencies
```
pip install -r requirements.txt
```
---
## Running Experiments
### 1️⃣ Full Fine-tuning
```
python train.py --config configs/imdb_full.yaml
```
### 2️⃣ LoRA Fine-tuning
```
python train.py --config configs/imdb_lora.yaml
```
### 3️⃣ BitFit Fine-tuning
```
python train.py --config configs/imdb_BitFit.yaml
```
### 4️⃣ Evaluation
After training, run evaluation:
```
python my_evaluate.py --config configs/imdb_BitFit.yaml 
# or imdb_full.yaml / imdb_lora.yaml
```
---
## Results
The following table summarizes the evaluation results of three fine-tuning strategies (Full, LoRA, BitFit) on the **IMDB sentiment classification dataset** using **DistilBERT-base-uncased**.

| Method               | Trainable Params | Eval Loss  |  Accuracy  |  F1 Score  | Notes                                |
| :------------------- | :--------------: | :--------: | :--------: | :--------: | :----------------------------------- |
| **Full Fine-tuning** |    66M (100%)    |   0.3487   | **0.9166** | **0.9565** | Highest accuracy but most parameters |
| **LoRA**             |    ≈3M (~5%)     | **0.2645** |   0.8948   |   0.9445   | Strong performance with 5% params    |
| **BitFit**           |   ≈0.7M (~1%)    |   0.3214   |   0.8670   |   0.9288   | Lightest method, lower accuracy      |

---
## 📁 Repository Structure
```
F:/NUS_AIS/DSA4213/3_assignment/  
├── configs/  
│ ├── imdb_adapter.yaml # Adapter configuration
│ ├── imdb_BitFit.yaml # BitFit configuration  
│ ├── imdb_full.yaml # Full fine-tuning configuration  
│ ├── imdb_lora.yaml # LoRA configuration  
│ └── imdb_prompt.yaml # Prompt-tuning configuration  
│  
├── logs/  
│ └── eval/  
│ └── runs/ # Evaluation logs  
│ ├── Oct06_10-31-07_shy-personal-PC  
│ ├── Oct06_13-26-12_shy-personal-PC  
│ └── Oct06_13-42-45_shy-personal-PC  
│  
├── outputs/  
│ ├── imdb_BitFit/ # BitFit fine-tuned models outputs  
│ ├── imdb_full/ # Full fine-tuned models outputs  
│ └── imdb_lora/ # LoRA fine-tuned models outputs  
│  
├── data_prep.py # Data preprocessing script  
├── evaluate.py # Official evaluation script  
├── my_evaluate.py # Custom evaluation script  
├── train.py # Main training script  
├── requirements.txt # Python dependencies  
└── method.txt # methods description and run commands
```
---
Note: the outputs/ directory (training outputs and fine‑tuned model weights) is not included in this repository. To avoid inflating the repository and because Git is not suitable for storing large binary files, we do not upload model checkpoints, generated outputs, or full datasets to GitHub. Run the training script locally to create the outputs directory and save the trained models 

(for example: python train.py --config configs/imdb_lora.yaml).
