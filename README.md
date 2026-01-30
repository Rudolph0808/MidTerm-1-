# Explainable AI for Credit Card Fraud Detection

TC2038 - Midterm 1 Project  
Rodolfo Vega Dominguez - A01566896

## Project Overview

Explainable fraud detection system using SHAP with Random Forest and XGBoost ensemble.

## Repository Structure

```
repo/
  data/         (dataset - excluded from git)
  notebooks/
    01_training.ipynb
    02_shap_explanations.ipynb
  src/
    train.py
    predict.py
    explain.py
  results/
    figures/
    tables/
  README.md
  requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Place `creditcard.csv` in `data/` folder

## Usage

Train models:
```bash
python src/train.py
```

Generate predictions:
```bash
python src/predict.py
```

Generate SHAP explanations:
```bash
python src/explain.py
```

## Implemented Components

- Data loading and preprocessing
- Random Forest and XGBoost training
- Basic SHAP value computation
- Performance evaluation metrics
