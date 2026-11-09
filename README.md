# Credit Card Customer Churn

🚀 **My first machine-learning project!** This project explores how customer and account data can help predict credit-card churn. It brings together data preprocessing, model evaluation, and an interactive app—and it’s a first step in a bigger journey of learning by building.

The project trains an XGBoost classifier to estimate churn probabilities. A Streamlit app lets you upload customer data and explore its predictions.

## Features

- Preprocesses numeric and categorical customer features
- Trains an XGBoost classification pipeline
- Evaluates predictions using stratified five-fold cross-validation
- Saves the trained pipeline for reuse
- Provides a Streamlit app for scoring customer data from a CSV

## Project structure

```text
.
├── app.py
├── train.py
├── pixi.toml
├── pixi.lock
├── data/
│   └── BankChurners.csv
├── model/
│   └── churn_model.joblib
└── notebooks/
    └── creditcardfraud.ipynb
```

## Requirements

This project uses [Pixi](https://pixi.sh/) to manage its environment and dependencies.

Dependencies specified in `pixi.toml`:

- Python `3.12.*`
- pandas `>=3.1.0,<4`
- scikit-learn `>=1.9.1,<2`
- NumPy `>=2.5.4,<3`
- seaborn `>=0.13.2,<0.14`
- XGBoost `>=3.4.2,<4`
- ipykernel `*`
- joblib `>=1.6.0,<2`
- Streamlit `>=1.67.0,<2`

## Getting started

Install the project environment from the repository root:

```bash
pixi install
```

## Train and evaluate the model

```bash
pixi run train
```

The training script reads `data/BankChurners.csv`, preprocesses the features, and trains an XGBoost model. It prints a classification report and ROC-AUC score based on out-of-fold predictions from five-fold stratified cross-validation, then saves the trained pipeline to `model/churn_model.joblib`.

The script also fits the pipeline on the full dataset before saving it. The cross-validation results are useful for evaluation, but this project does not currently use a separate holdout test set.

## Run the Streamlit app

From the repository root:

```bash
pixi run start-app
```

Upload a CSV containing the required feature columns. The app displays the uploaded data with a `churn_probability` column. This value is between 0 and 1; multiply by 100 to read it as a percentage.

The model’s positive class is `Attrited Customer`. The input CSV does not need an `Attrition_Flag` column.

### Required input columns

**Numeric features**

- `Customer_Age`
- `Dependent_count`
- `Months_on_book`
- `Total_Relationship_Count`
- `Months_Inactive_12_mon`
- `Contacts_Count_12_mon`
- `Credit_Limit`
- `Total_Revolving_Bal`
- `Total_Amt_Chng_Q4_Q1`
- `Total_Trans_Amt`
- `Total_Trans_Ct`
- `Total_Ct_Chng_Q4_Q1`
- `Avg_Utilization_Ratio`

**Categorical features**

- `Income_Category`
- `Card_Category`

## How the model works

- Numeric missing values are imputed with the median, then standardized.
- Missing categorical values are imputed with the most frequent value.
- `Income_Category` is one-hot encoded.
- `Card_Category` is ordinal-encoded in this order: Blue, Silver, Gold, Platinum.
- The XGBoost classifier uses class weighting based on the target class distribution.

The fitted preprocessing steps are included in the saved pipeline, so the app applies the same transformations used during training.

## Data

Training data is expected at `data/BankChurners.csv`. It must contain the feature columns listed above and an `Attrition_Flag` target column. The target is mapped to `1` for `Attrited Customer` and `0` for other values.

Before redistributing the dataset, check its source and license and make sure you have permission to share it.

## Limitations

- Cross-validation provides an estimate of model performance, but there is no independent holdout test evaluation yet.
- Predictions are estimates based on the available training data and features; they should not be treated as definitive customer outcomes.
- Uploaded CSV files must include all required feature columns.
