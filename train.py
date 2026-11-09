from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "BankChurners.csv"
MODEL_PATH = PROJECT_DIR / "model" / "churn_model.joblib"


NUMERIC_FEATURES = ['Customer_Age',
                     'Dependent_count', 'Months_on_book',
                       'Total_Relationship_Count',
                         'Months_Inactive_12_mon',
                           'Contacts_Count_12_mon', 'Credit_Limit',
                             'Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1',
                               'Total_Trans_Amt', 'Total_Trans_Ct',
                                 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

CATEGORICAL_FEATURES = ["Income_Category"]

CARD_CATEGORY = ["Card_Category"]  # Ordinal categorical feature
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + CARD_CATEGORY


def main():
    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["Attrition_Flag"]
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df.dropna(subset=["Attrition_Flag"])
    X = df[FEATURES].copy()
    y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "card category",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=[["Blue", "Silver", "Gold", "Platinum"]],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CARD_CATEGORY,
            ),
        ]
    )

    positive_weight = (y == 0).sum() / (y == 1).sum()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    scale_pos_weight=positive_weight,
                    max_depth=5,
                    learning_rate=0.03,
                    n_estimators=100,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    model.fit(X, y)
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    predictions = cross_val_predict(model, X, y, cv=cv_stratified)
    probabilities = cross_val_predict(model, X, y, cv=cv_stratified, method='predict_proba')[:, 1]

    # Evaluate
    print("Classification Report:")
    print(classification_report(y, predictions, target_names=['Existing', 'Churned']))

    print(f"AUC-ROC: {roc_auc_score(y, probabilities):.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()