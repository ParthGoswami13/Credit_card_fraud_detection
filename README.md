# Credit Card Fraud Detection

**Project type:** Data Science / Machine Learning

**Short description**
This project detects fraudulent credit card transactions using machine learning. It includes data exploration, preprocessing, feature engineering, model training, evaluation, and a short guide to reproduce results. The main work is in the Jupyter notebook `Project_Credit_Card_Fraud_Detection.ipynb`.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
5. [Modeling](#modeling)
6. [How to run](#how-to-run)
7. [Repository structure](#repository-structure)
8. [Requirements](#requirements)
9. [License & Contact](#license--contact)

---

## Project Overview

This project builds a fraud detection pipeline on transactional data. The goal is to identify fraudulent transactions (rare class) while keeping false positives low. The notebook walks step-by-step from data loading to model selection and evaluation.

## Dataset

* The dataset used is the commonly used Credit Card Fraud Detection dataset hosted on Kaggle.

**Kaggle dataset page:** [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

* Typical columns: `Time`, `V1`..`V28` (anonymized PCA features), `Amount`, `Class` (0 = normal, 1 = fraud).
* The dataset is **not included** in this repository (Kaggle dataset terms & size). Download it and place the CSV in the `data/` folder with the exact filename `data/creditcard.csv`.

### Downloading the dataset (recommended)

If you have the Kaggle CLI set up, you can download the dataset directly into the `data/` folder with:

```bash
# create data folder if it doesn't exist
mkdir -p data

# download and unzip (requires kaggle CLI and authentication)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip

# After download you should have data/creditcard.csv
```

If you don't have the Kaggle CLI, go to the Kaggle dataset page (link above), download the `creditcard.csv` manually, and put it into the `data/` folder.

## Exploratory Data Analysis (EDA)

Key EDA steps included in the notebook:

* Inspecting basic statistics and class balance.
* Visualizing distribution of `Amount` and `Time`.
* Checking correlations and feature distributions.
* Visual checks for imbalance and outliers.

## Preprocessing & Feature Engineering

This project demonstrates common preprocessing steps:

* Handling class imbalance (options shown: under-sampling, over-sampling (SMOTE), and class weights).
* Scaling the `Amount` feature (StandardScaler / RobustScaler) and leaving PCA features as-is.
* Train / validation / test splitting with stratification to preserve class ratio.
* Optional feature selection and dimensionality reduction demonstrated in the notebook.

## Modeling

Models used (examples provided in the notebook):

* Logistic Regression (with class weights)
* Random Forest
* XGBoost / LightGBM (if available)
* Baseline: Dummy classifier

The notebook shows hyperparameter tuning examples (GridSearchCV / RandomizedSearchCV) and cross-validation.

## How to run

1. Clone the repo (or copy files into your local project folder).
2. Download the dataset from Kaggle and place it into `data/creditcard.csv` (create `data/` if missing). See the **Dataset** section above for the `kaggle` CLI command.
3. Create and activate a Python environment (see **Requirements** below).
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the Jupyter notebook:

```bash
jupyter notebook Project_Credit_Card_Fraud_Detection.ipynb
```

6. Follow cells in the notebook. You can run end-to-end or run section-by-section.

## Repository structure

```
project-root/
├─ data/                      # dataset (not included by default) -> place creditcard.csv here
│  └─ creditcard.csv
├─ Project_Credit_Card_Fraud_Detection.ipynb
├─ README.md                
```

Adjust paths in the notebook if you move files.

## Requirements

A minimal `requirements.txt` might contain:

```
numpy
pandas
scikit-learn
matplotlib
seaborn
imbalanced-learn
xgboost
jupyter
joblib
kaggle
```

(If you don't use XGBoost, you can remove it. If you prefer conda, create an `environment.yml`.)

## Notes & Next steps

* You can improve this project by adding a small web demo (Flask or Streamlit) that lets a user upload a transaction and get a prediction.
* Consider experimenting with cost-sensitive learning, anomaly detection methods, or feature attribution (SHAP) to explain model decisions.


---

**Short conclusion:**
This repository contains a reproducible pipeline for credit card fraud detection with clear steps for data handling, modeling, evaluation, and reproducibility. Use the notebook to explore, train, and evaluate models; then add improvements as desired.


