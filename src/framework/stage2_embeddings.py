from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .io_utils import ensure_parent_dir, read_csv_smart, save_json


def load_dataset(input_csv: str, text_column: str, label_column: str) -> pd.DataFrame:
    df = read_csv_smart(input_csv)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    df = df[[text_column, label_column]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna()
    df = df[df["text"].str.len() > 10]
    df = df[df["label"].str.len() > 0]
    return df


def get_models(seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM (Linear)": SVC(kernel="linear", random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }


def run_stage2_embeddings(
    input_csv: str,
    output_dir: str,
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    model_name: str = "dominguesm/legal-bert-base-cased-ptbr",
    k_folds: int = 3,
    seed: int = 42,
) -> dict:
    np.random.seed(seed)

    df = load_dataset(input_csv, text_column=text_column, label_column=label_column)
    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])

    embedder = SentenceTransformer(model_name)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    by_model = {name: [] for name in get_models(seed)}
    aggregate = {name: {"preds": [], "trues": []} for name in get_models(seed)}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, df["label_encoded"]), start=1):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        x_train = embedder.encode(train_df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
        x_test = embedder.encode(test_df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
        y_train = train_df["label_encoded"].values
        y_test = test_df["label_encoded"].values

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        for name, model in get_models(seed).items():
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            metrics = {
                "fold": fold,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            }
            by_model[name].append(metrics)
            aggregate[name]["preds"].extend(np.asarray(y_pred).tolist())
            aggregate[name]["trues"].extend(np.asarray(y_test).tolist())

    report = {
        "embedding_model": model_name,
        "classes": encoder.classes_.tolist(),
        "models": {},
    }

    for name, metrics in by_model.items():
        f1_values = [m["f1"] for m in metrics]
        report["models"][name] = {
            "fold_metrics": metrics,
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "classification_report": classification_report(
                aggregate[name]["trues"],
                aggregate[name]["preds"],
                target_names=encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        }

    output_path = Path(output_dir) / "stage2_embeddings_results.json"
    ensure_parent_dir(output_path)
    save_json(report, output_path)
    print(f"\nResultados salvos em: {output_path}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 2 (embeddings) - BERT embeddings + classificadores")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    parser.add_argument("--k-folds", type=int, default=3)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_stage2_embeddings(
        input_csv=args.input,
        output_dir=args.output_dir,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        k_folds=args.k_folds,
    )


if __name__ == "__main__":
    main()
