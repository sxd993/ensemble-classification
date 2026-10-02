"""Классификация Census Income методами ансамблей."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier

ADULT_COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
TARGET = "income"


def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, names=ADULT_COLUMNS, header=None, skipinitialspace=True, comment="|", na_values=["?", " ?"])
    frame = frame.dropna(subset=[TARGET]).copy()
    frame[TARGET] = frame[TARGET].astype(str).str.strip().str.rstrip(".")
    for column in frame.columns:
        if column != TARGET:
            frame[column] = frame[column].fillna(frame[column].median() if pd.api.types.is_numeric_dtype(frame[column]) else frame[column].mode().iloc[0])
    return frame


def split_encode(frame: pd.DataFrame, train_fraction: float, random_state: int):
    indices = np.random.default_rng(random_state).permutation(len(frame))
    cut = int(len(frame) * train_fraction)
    train, test = frame.iloc[indices[:cut]], frame.iloc[indices[cut:]]
    x_train = pd.get_dummies(train.drop(columns=TARGET), dtype=float)
    x_test = pd.get_dummies(test.drop(columns=TARGET), dtype=float).reindex(columns=x_train.columns, fill_value=0)
    return x_train, x_test, train[TARGET], test[TARGET]


def build_model(technique: str, n_estimators: int, max_depth: int | None, max_samples: float, learning_rate: float, random_state: int):
    base = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    if technique == "bagging":
        return BaggingClassifier(estimator=base, n_estimators=n_estimators, max_samples=max_samples, random_state=random_state, n_jobs=-1)
    if technique == "random_forest":
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state), n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


def run(data_path: str, technique: str, n_estimators: int, train_fraction: float = .8, max_depth: int | None = 8, max_samples: float = 1.0, learning_rate: float = 1.0, random_state: int = 42) -> dict:
    x_train, x_test, y_train, y_test = split_encode(load_data(data_path), train_fraction, random_state)
    model = build_model(technique, n_estimators, max_depth, max_samples, learning_rate, random_state)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, prediction, pos_label=">50K", average="binary", zero_division=0)
    return {"technique": technique, "n_estimators": n_estimators, "train_fraction": train_fraction, "train_size": len(y_train), "test_size": len(y_test), "accuracy": accuracy_score(y_test, prediction), "precision": precision, "recall": recall, "f1": f1, "max_depth": max_depth, "max_samples": max_samples, "learning_rate": learning_rate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ансамблевая классификация Census Income")
    parser.add_argument("--data", required=True)
    parser.add_argument("--technique", choices=["bagging", "random_forest", "boosting"], default="random_forest")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-samples", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1: parser.error("--train-fraction должен быть в интервале (0, 1)")
    result = run(args.data, args.technique, args.n_estimators, args.train_fraction, args.max_depth, args.max_samples, args.learning_rate, args.random_state)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
