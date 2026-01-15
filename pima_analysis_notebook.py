import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TARGET_COL = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
RANDOM_STATE = 42

# Set dataset path for Jupyter use
DATA_PATH = "data/diabetes.csv"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}. Put Kaggle diabetes.csv there or update DATA_PATH."
        )
    return pd.read_csv(path)


def replace_zeros_with_nan(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    return df


def build_pipeline(model, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def get_score_vector(model, x_test: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return None


def evaluate_model(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> dict:
    pipeline.fit(x_train, y_train)
    preds = pipeline.predict(x_test)
    scores = get_score_vector(pipeline, x_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, scores)

    print(f"\n== {name} ==")
    print(classification_report(y_test, preds, zero_division=0))
    print(metrics)
    return metrics


def cross_validate_auc(name: str, pipeline: Pipeline, x, y) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    try:
        auc_scores = cross_val_score(pipeline, x, y, scoring="roc_auc", cv=cv)
        print(f"{name} CV AUC: mean={auc_scores.mean():.4f} std={auc_scores.std():.4f}")
    except Exception as exc:
        print(f"{name} CV AUC failed: {exc}")


# Load and inspect
_df = load_data(DATA_PATH)
print("Shape:", _df.shape)
print(_df.head(5))

_zero_counts = {}
for _col in ZERO_AS_MISSING:
    if _col in _df.columns:
        _zero_counts[_col] = int((_df[_col] == 0).sum())
if _zero_counts:
    print("Zero counts (treated as missing):", _zero_counts)

_df = replace_zeros_with_nan(_df, ZERO_AS_MISSING)
_missing = _df.isna().sum()
print("Missing values after replacement:\n", _missing[_missing > 0])

if TARGET_COL not in _df.columns:
    raise ValueError(f"Missing target column: {TARGET_COL}")

_feature_cols = [c for c in _df.columns if c != TARGET_COL]
_x = _df[_feature_cols]
_y = _df[TARGET_COL]

_x_train, _x_test, _y_train, _y_test = train_test_split(
    _x, _y, test_size=0.2, stratify=_y, random_state=RANDOM_STATE
)

_models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
    "RandomForest": (
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
        False,
    ),
    "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    "SVM": (SVC(kernel="rbf", probability=True, class_weight="balanced"), True),
    "KNN": (KNeighborsClassifier(n_neighbors=15), True),
}

_results = []
for _name, (_model, _scale) in _models.items():
    _pipeline = build_pipeline(_model, _scale)
    cross_validate_auc(_name, _pipeline, _x, _y)
    _metrics = evaluate_model(_name, _pipeline, _x_train, _x_test, _y_train, _y_test)
    _results.append(_metrics)

_best = None
for _item in _results:
    if "roc_auc" not in _item:
        continue
    if _best is None or _item["roc_auc"] > _best["roc_auc"]:
        _best = _item

if _best:
    print(f"\nBest model by test ROC AUC: {_best['model']} ({_best['roc_auc']:.4f})")