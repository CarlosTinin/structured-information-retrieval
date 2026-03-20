from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .io_utils import ensure_parent_dir, load_json_flexible, read_csv_smart, save_json


@dataclass
class FineTuneConfig:
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 5e-5
    epochs: int = 8
    k_folds: int = 3
    seed: int = 42


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = [str(t) for t in texts]
        self.labels = [int(l) for l in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


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


def _evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            y_pred = torch.argmax(outputs.logits, dim=1)
            preds.extend(y_pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    return {
        "accuracy": float(accuracy_score(trues, preds)),
        "precision": float(precision_score(trues, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(trues, preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(trues, preds, average="weighted", zero_division=0)),
        "preds": preds,
        "trues": trues,
    }


def _train_fold(model, train_loader, val_loader, class_weights, config: FineTuneConfig, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = -1.0
    patience, wait = 3, 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())

        val_metrics = _evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Loss={epoch_loss / max(1, len(train_loader)):.4f} | "
            f"Val_F1={val_metrics['f1']:.4f}"
        )
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    return model


def run_stage2_finetune(
    input_csv: str,
    output_root: str = "output",
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    target_labels: tuple[str, ...] = ("condenação", "extinto", "absolvição"),
    embeddings_results_json: str = "output/stage2_embeddings_results.json",
    config: FineTuneConfig | None = None,
) -> dict:
    config = config or FineTuneConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if config.k_folds > min_class_size:
        print(f"[Aviso] Ajustando k_folds de {config.k_folds} para {min_class_size} (classe minoritária).")
        config.k_folds = min_class_size

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    splitter = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    fold_metrics = []
    all_preds, all_trues = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, df["label_encoded"]), start=1):
        print(f"\n--- Fold {fold_idx}/{config.k_folds} ---")
        train_fold = df.iloc[train_idx].copy()
        test_fold = df.iloc[test_idx].copy()

        train_data, val_data = train_test_split(
            train_fold,
            test_size=0.2,
            stratify=train_fold["label_encoded"],
            random_state=config.seed,
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_data["label_encoded"]),
            y=train_data["label_encoded"],
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        train_loader = DataLoader(
            TextDataset(train_data["text"].tolist(), train_data["label_encoded"].tolist(), tokenizer, config.max_length),
            batch_size=config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TextDataset(val_data["text"].tolist(), val_data["label_encoded"].tolist(), tokenizer, config.max_length),
            batch_size=config.batch_size,
        )
        test_loader = DataLoader(
            TextDataset(test_fold["text"].tolist(), test_fold["label_encoded"].tolist(), tokenizer, config.max_length),
            batch_size=config.batch_size,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=len(encoder.classes_),
            ignore_mismatched_sizes=True,
        ).to(device)

        model = _train_fold(model, train_loader, val_loader, class_weights, config, device)
        metrics = _evaluate(model, test_loader, device)
        fold_metrics.append({k: metrics[k] for k in ["accuracy", "precision", "recall", "f1"]})
        all_preds.extend(metrics["preds"])
        all_trues.extend(metrics["trues"])

    summary = {
        "model_name": config.model_name,
        "classes": encoder.classes_.tolist(),
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "classification_report": classification_report(
            all_trues,
            all_preds,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_trues, all_preds).tolist(),
    }

    output_path = Path(output_root) / "stage2_finetune_results.json"
    ensure_parent_dir(output_path)
    save_json(summary, output_path)

    # Gera tabela separada de comparação em output/tables (formato LaTeX)
    fine_row = {
        "Abordagem": "BERT fine-tuned",
        "Modelo": config.model_name,
        "Accuracy": float(summary["mean_accuracy"]),
        "Precision": float(np.mean([m["precision"] for m in fold_metrics])) if fold_metrics else 0.0,
        "Recall": float(np.mean([m["recall"] for m in fold_metrics])) if fold_metrics else 0.0,
        "F1": float(summary["mean_f1"]),
    }

    rows = [fine_row]
    embed_path = Path(embeddings_results_json)
    if embed_path.exists():
        try:
            emb_results = load_json_flexible(str(embed_path))
            emb_models = emb_results.get("models", {}) if isinstance(emb_results, dict) else {}
            if emb_models:
                best_name, best_data = max(
                    emb_models.items(),
                    key=lambda kv: float(kv[1].get("mean_f1", 0.0)),
                )
                fold_m = best_data.get("fold_metrics", [])
                rows.append(
                    {
                        "Abordagem": "BERT encoder + clássico",
                        "Modelo": best_name,
                        "Accuracy": float(np.mean([m.get("accuracy", 0.0) for m in fold_m])) if fold_m else 0.0,
                        "Precision": float(np.mean([m.get("precision", 0.0) for m in fold_m])) if fold_m else 0.0,
                        "Recall": float(np.mean([m.get("recall", 0.0) for m in fold_m])) if fold_m else 0.0,
                        "F1": float(best_data.get("mean_f1", 0.0)),
                    }
                )
        except Exception as exc:
            print(f"[Aviso] Não foi possível ler resultados de embeddings em {embed_path}: {exc}")

    comparison_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    tables_dir = Path(output_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    render_df = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        render_df[col] = render_df[col].map(lambda x: f"{x:.4f}")

    table_tex_path = tables_dir / "table_stage2_finetune_comparison.tex"
    latex_table = render_df.to_latex(index=False, escape=True)
    table_tex_path.write_text(latex_table, encoding="utf-8")

    print(f"\nResultados salvos em: {output_path}")
    print(f"Tabela de comparação (LaTeX) salva em: {table_tex_path}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 2 (fine-tuning) - BERT")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--target-labels", default="condenação,extinto,absolvição")
    parser.add_argument("--embeddings-results-json", default="output/stage2_embeddings_results.json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FineTuneConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_folds=args.k_folds,
    )
    target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
    run_stage2_finetune(
        input_csv=args.input,
        output_root=args.output_root,
        text_column=args.text_column,
        label_column=args.label_column,
        target_labels=target_labels,
        embeddings_results_json=args.embeddings_results_json,
        config=cfg,
    )


if __name__ == "__main__":
    main()
