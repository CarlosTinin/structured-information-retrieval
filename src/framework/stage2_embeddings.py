from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .io_utils import ensure_parent_dir, read_csv_smart, save_json

try:
    import xgboost as xgb
except Exception:
    xgb = None


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
    models = {
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
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        print("[Aviso] XGBoost indisponível no ambiente (libomp ausente). Modelo será ignorado.")

    return models


def run_stage2_embeddings(
    input_csv: str,
    output_root: str = "output",
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    model_name: str = "dominguesm/legal-bert-base-cased-ptbr",
    k_folds: int = 3,
    seed: int = 42,
    target_labels: tuple[str, ...] = ("condenação", "extinto", "absolvição"),
) -> dict:
    np.random.seed(seed)

    df = load_dataset(input_csv, text_column=text_column, label_column=label_column)

    if target_labels:
        target_set = {x.strip().lower() for x in target_labels}
        before = len(df)
        df = df[df["label"].isin(target_set)].copy()
        print(f"Filtrando classes-alvo: {sorted(target_set)} | {before} -> {len(df)} documentos")

    class_counts = df["label"].value_counts()
    if class_counts.empty:
        raise ValueError("Nenhum documento disponível após filtragem de classes-alvo.")

    min_class_size = int(class_counts.min())
    if min_class_size < 2:
        raise ValueError(
            "Pelo menos uma classe tem menos de 2 exemplos após filtragem. "
            "Não é possível executar validação cruzada estratificada."
        )

    if k_folds > min_class_size:
        print(f"[Aviso] Ajustando k_folds de {k_folds} para {min_class_size} (classe minoritária).")
        k_folds = min_class_size

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
            try:
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
            except Exception as exc:
                print(f"[Aviso] Modelo {name} falhou no fold {fold}: {exc}")

    report = {
        "embedding_model": model_name,
        "classes": encoder.classes_.tolist(),
        "models": {},
    }

    for name, metrics in by_model.items():
        if not metrics:
            continue
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

    # Resumo para terminal e tabela
    rows = []
    for name, metrics in by_model.items():
        if not metrics:
            continue
        rows.append(
            {
                "Modelo": name,
                "Accuracy": float(np.mean([m["accuracy"] for m in metrics])),
                "Precision": float(np.mean([m["precision"] for m in metrics])),
                "Recall": float(np.mean([m["recall"] for m in metrics])),
                "F1": float(np.mean([m["f1"] for m in metrics])),
            }
        )
    if not rows:
        raise RuntimeError("Nenhum modelo concluiu avaliação com sucesso.")
    metrics_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    print("\nResumo de desempenho (média dos folds):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Exportar tabela LaTeX
    table_path = Path(output_root) / "tables" / "table.tex"
    ensure_parent_dir(table_path)
    latex_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}")
    latex_table = latex_df.to_latex(index=False, escape=True)
    table_path.write_text(latex_table, encoding="utf-8")

    # Exportar matrizes de confusão por modelo
    images_dir = Path(output_root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for name in by_model.keys():
        if not by_model[name]:
            continue
        y_true = aggregate[name]["trues"]
        y_pred = aggregate[name]["preds"]
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_,
            cbar=False,
        )
        plt.title(f"Matriz de Confusão - {name}")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.tight_layout()

        model_slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        image_path = images_dir / f"matriz_confusao_{model_slug}.png"
        plt.savefig(image_path, dpi=200)
        plt.close()

    output_path = Path(output_root) / "stage2_embeddings_results.json"
    ensure_parent_dir(output_path)
    save_json(report, output_path)
    print(f"\nResultados salvos em: {output_path}")
    print(f"Tabela LaTeX salva em: {table_path}")
    print(f"Imagens salvas em: {images_dir}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 2 (embeddings) - BERT embeddings + classificadores")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--target-labels", default="condenação,extinto,absolvição")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
    run_stage2_embeddings(
        input_csv=args.input,
        output_root=args.output_root,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        k_folds=args.k_folds,
        target_labels=target_labels,
    )


if __name__ == "__main__":
    main()
