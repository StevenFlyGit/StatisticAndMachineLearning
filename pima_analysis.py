import argparse
import os
from typing import Dict, Tuple

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

# 类似于Java中的静态常量，不可修改
TARGET_COL = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"] 
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}. Put Kaggle diabetes.csv there or pass --data."
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


def evaluate_model(name: str, pipeline: Pipeline, x_train, x_test, y_train, y_test) -> Dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pima Indians Diabetes ML analysis")
    parser.add_argument("--data", default="data/diabetes.csv", help="Path to diabetes.csv")
    parser.add_argument("--save", action="store_true", help="Save best model to models/")
    args = parser.parse_args()

    df = load_data(args.data)
    print("Shape:", df.shape)
    print(df.head(5))

    zero_counts = {}
    for col in ZERO_AS_MISSING: # ZERO_AS_MISSING表示将0视为缺失值的列名列表，这些列的中的0值通常会被认为是不符合逻辑的值，需要被替换为缺失值
        if col in df.columns:
            print("df[col]:", col)
            print("type df[col]:", type(df[col]))
            print(df[col])
            zero_counts[col] = int((df[col] == 0).sum())
    if zero_counts:
        print("Zero counts (treated as missing):", zero_counts)

    df = replace_zeros_with_nan(df, ZERO_AS_MISSING)
    missing = df.isna().sum()
    print("c:", type(missing))
    print("Missing values after replacement:\n", missing[missing > 0])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    x = df[feature_cols]
    y = df[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models: Dict[str, Tuple[object, bool]] = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"),
            False,
        ),
        "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False),
        "SVM": (SVC(kernel="rbf", probability=True, class_weight="balanced"), True),
        "KNN": (KNeighborsClassifier(n_neighbors=15), True),
    }

    results = []
    for name, (model, scale) in models.items():
        pipeline = build_pipeline(model, scale)
        cross_validate_auc(name, pipeline, x, y)
        metrics = evaluate_model(name, pipeline, x_train, x_test, y_train, y_test)
        results.append(metrics)

    best = None
    for item in results:
        if "roc_auc" not in item:
            continue
        if best is None or item["roc_auc"] > best["roc_auc"]:
            best = item

    if best:
        print(f"\nBest model by test ROC AUC: {best['model']} ({best['roc_auc']:.4f})")

    if args.save and best:
        os.makedirs("models", exist_ok=True)
        best_name = best["model"]
        best_model, best_scale = models[best_name]
        best_pipeline = build_pipeline(best_model, best_scale)
        best_pipeline.fit(x, y)
        import joblib

        out_path = os.path.join("models", "best_model.joblib")
        joblib.dump(best_pipeline, out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()