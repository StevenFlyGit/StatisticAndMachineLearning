# Pima Indians Diabetes - ML Analysis

This repo includes a simple ML analysis script for the Kaggle Pima Indians Diabetes dataset. - Mainly for course project

Steps:
1) Download `diabetes.csv` from Kaggle and place it at `data/diabetes.csv`.
2) Install dependencies: `pip install -r requirements.txt`.
3) Run the analysis: `python pima_analysis.py --data data/diabetes.csv`.

The script prints basic EDA, treats zeros as missing for selected columns, trains multiple models,
reports metrics, and optionally saves the best model with `--save`.
