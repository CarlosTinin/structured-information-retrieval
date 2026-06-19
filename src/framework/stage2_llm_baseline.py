"""Stage 2 – LLM zero-shot and few-shot baseline for penal-merit classification.

Uses Gemini Pro 2.5 to classify legal documents into condenação/absolvição/extinto,
evaluated on the SAME stratified K-fold test splits as stage2_embeddings for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from .io_utils import ensure_parent_dir, read_csv_smart, save_json


# ---------------------------------------------------------------------------
# API key resolution (reuses same pattern as stage3)
# ---------------------------------------------------------------------------

def _read_env_file_key(env_path: Path, key: str) -> str:
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _resolve_api_key(key_name: str) -> str:
    api_key = os.getenv(key_name, "").strip()
    if api_key:
        return api_key
    for env_path in (Path(".env"), Path(__file__).resolve().parents[2] / ".env"):
        key = _read_env_file_key(env_path, key_name)
        if key:
            os.environ[key_name] = key
            return key
    return ""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _load_base_prompt(prompt_file: str) -> str:
    return Path(prompt_file).read_text(encoding="utf-8")


def _build_zero_shot_prompt(base_prompt: str, text: str) -> str:
    return base_prompt.replace("{texto}", text)


def _build_few_shot_prompt(
    base_prompt: str,
    text: str,
    examples: list[dict[str, str]],
) -> str:
    """Prepend labeled examples before the target document."""
    examples_block = "\n\n".join(
        f"--- Exemplo ({ex['label']}) ---\n{ex['text'][:2000]}\n"
        f'Resposta: {{"decisao": "{ex["label"]}"}}'
        for ex in examples
    )
    # Insert examples between instructions and the target document
    parts = base_prompt.split("{texto}")
    if len(parts) == 2:
        prompt = (
            parts[0]
            + "\nExemplos de referência:\n\n"
            + examples_block
            + "\n\nAgora classifique o documento abaixo:\n\n"
            + text
        )
    else:
        prompt = base_prompt.replace("{texto}", text)
    return prompt


def _select_few_shot_examples(
    train_df: pd.DataFrame,
    target_labels: list[str],
    n_per_class: int = 3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Select n_per_class examples from training set, one per class."""
    rng = random.Random(seed)
    examples = []
    for label in sorted(target_labels):
        candidates = train_df[train_df["label"] == label]
        if candidates.empty:
            continue
        # Pick shortest documents as examples (more likely to fit context)
        sorted_candidates = candidates.sort_values(
            by="text", key=lambda s: s.str.len()
        )
        pool = sorted_candidates.head(min(10, len(sorted_candidates)))
        selected = pool.sample(n=min(n_per_class, len(pool)), random_state=seed)
        for _, row in selected.iterrows():
            examples.append({"text": row["text"], "label": row["label"]})
    rng.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

def _parse_llm_response(response_text: str, valid_labels: set[str]) -> str | None:
    """Extract the decision label from LLM JSON response."""
    # Try JSON parse
    try:
        obj = json.loads(response_text)
        label = obj.get("decisao", "").strip().lower()
        if label in valid_labels:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex
    match = re.search(r'"decisao"\s*:\s*"([^"]+)"', response_text)
    if match:
        label = match.group(1).strip().lower()
        if label in valid_labels:
            return label

    # Last resort: check if any valid label appears as standalone word
    for label in valid_labels:
        if label in response_text.lower():
            return label

    return None


def _classify_document(
    model: Any,
    prompt: str,
    valid_labels: set[str],
    max_retries: int = 3,
    retry_backoff: float = 5.0,
    request_timeout: int = 120,
) -> str | None:
    """Send prompt to Gemini and return parsed label."""
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"},
                request_options={"timeout": request_timeout},
            )
            # Extract text from response
            text = ""
            try:
                text = response.text
            except Exception:
                candidates = getattr(response, "candidates", None)
                if candidates:
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        parts = getattr(content, "parts", None) if content else None
                        if parts:
                            for part in parts:
                                t = getattr(part, "text", None)
                                if t:
                                    text += t

            label = _parse_llm_response(text, valid_labels)
            return label

        except Exception as exc:
            if attempt < max_retries:
                wait = retry_backoff * attempt
                print(f"    [Retry {attempt}/{max_retries}] Error: {exc}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [FAILED] All retries exhausted: {exc}")
                return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage2_llm_baseline(
    input_csv: str,
    prompt_file: str,
    output_root: str = "output",
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    model_name: str = "gemini-2.5-pro",
    k_folds: int = 3,
    seed: int = 42,
    target_labels: tuple[str, ...] = ("condenação", "extinto", "absolvição"),
    n_few_shot_per_class: int = 3,
    sleep_between_requests: float = 1.0,
    api_key_env: str = "GEMINI_API_KEY",
    max_retries: int = 3,
    request_timeout: int = 120,
) -> dict:
    """Run zero-shot and few-shot LLM classification using same K-fold splits."""

    np.random.seed(seed)

    # --- Load dataset (same logic as stage2_embeddings) ---
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

    target_set = {x.strip().lower() for x in target_labels}
    before = len(df)
    df = df[df["label"].isin(target_set)].copy()
    df = df.reset_index(drop=True)
    print(f"Filtrando classes-alvo: {sorted(target_set)} | {before} -> {len(df)} documentos")

    class_counts = df["label"].value_counts()
    min_class_size = int(class_counts.min())
    if min_class_size < 2:
        raise ValueError("Classe com menos de 2 exemplos. Impossível executar validação cruzada.")

    if k_folds > min_class_size:
        print(f"[Aviso] Ajustando k_folds de {k_folds} para {min_class_size}.")
        k_folds = min_class_size

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])

    # --- Setup Gemini ---
    api_key = _resolve_api_key(api_key_env)
    if not api_key:
        raise ValueError(f"API key não encontrada. Defina {api_key_env}.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "Pacote 'google-generativeai' necessário. Instale: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    base_prompt = _load_base_prompt(prompt_file)
    valid_labels = target_set

    # --- Stratified K-Fold (same seed as embeddings) ---
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    modes = ["zero_shot", "few_shot"]
    aggregate = {mode: {"preds": [], "trues": []} for mode in modes}
    by_fold = {mode: [] for mode in modes}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, df["label_encoded"]), start=1):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}/{k_folds}")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # Select few-shot examples from training fold
        few_shot_examples = _select_few_shot_examples(
            train_df, sorted(target_set), n_per_class=n_few_shot_per_class, seed=seed + fold
        )
        print(f"  Few-shot examples selected: {len(few_shot_examples)}")

        for mode in modes:
            print(f"\n  --- Mode: {mode} ---")
            fold_preds = []
            fold_trues = []

            for i, (idx, row) in enumerate(test_df.iterrows()):
                if mode == "zero_shot":
                    prompt = _build_zero_shot_prompt(base_prompt, row["text"])
                else:
                    prompt = _build_few_shot_prompt(base_prompt, row["text"], few_shot_examples)

                pred_label = _classify_document(
                    model, prompt, valid_labels,
                    max_retries=max_retries,
                    request_timeout=request_timeout,
                )

                if pred_label is None:
                    # Default to majority class on failure (conservative)
                    pred_label = class_counts.idxmax()
                    print(f"    [{i+1}/{len(test_df)}] doc fallback -> {pred_label}")
                else:
                    print(f"    [{i+1}/{len(test_df)}] predicted: {pred_label} | true: {row['label']}")

                pred_encoded = encoder.transform([pred_label])[0]
                true_encoded = row["label_encoded"]

                fold_preds.append(pred_encoded)
                fold_trues.append(true_encoded)

                if sleep_between_requests > 0:
                    time.sleep(sleep_between_requests)

            # Fold metrics
            fold_metrics = {
                "fold": fold,
                "accuracy": float(accuracy_score(fold_trues, fold_preds)),
                "precision": float(precision_score(fold_trues, fold_preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(fold_trues, fold_preds, average="weighted", zero_division=0)),
                "f1": float(f1_score(fold_trues, fold_preds, average="weighted", zero_division=0)),
            }
            by_fold[mode].append(fold_metrics)
            aggregate[mode]["preds"].extend(fold_preds)
            aggregate[mode]["trues"].extend(fold_trues)

    # --- Compute final results ---
    report: dict[str, Any] = {
        "llm_model": model_name,
        "classes": encoder.classes_.tolist(),
        "k_folds": k_folds,
        "seed": seed,
        "n_few_shot_per_class": n_few_shot_per_class,
        "models": {},
    }

    display_names = {
        "zero_shot": f"Gemini 2.5 Pro (zero-shot)",
        "few_shot": f"Gemini 2.5 Pro (few-shot, {n_few_shot_per_class}/class)",
    }

    rows_for_table = []

    for mode in modes:
        f1_values = [m["f1"] for m in by_fold[mode]]
        model_report = {
            "fold_metrics": by_fold[mode],
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "classification_report": classification_report(
                aggregate[mode]["trues"],
                aggregate[mode]["preds"],
                target_names=encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        }
        report["models"][display_names[mode]] = model_report

        rows_for_table.append({
            "Model": display_names[mode],
            "Accuracy": float(np.mean([m["accuracy"] for m in by_fold[mode]])),
            "Precision": float(np.mean([m["precision"] for m in by_fold[mode]])),
            "Recall": float(np.mean([m["recall"] for m in by_fold[mode]])),
            "F1": float(np.mean([m["f1"] for m in by_fold[mode]])),
        })

    # --- Print summary ---
    metrics_df = pd.DataFrame(rows_for_table).sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 60)
    print("LLM Baseline - Resumo de desempenho (média dos folds)")
    print("=" * 60)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Export LaTeX table ---
    table_path = Path(output_root) / "tables" / "table_llm_baseline.tex"
    ensure_parent_dir(table_path)
    latex_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}")
    table_path.write_text(latex_df.to_latex(index=False, escape=True), encoding="utf-8")

    # --- Export confusion matrices ---
    images_dir = Path(output_root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        y_true = aggregate[mode]["trues"]
        y_pred = aggregate[mode]["preds"]
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_, yticklabels=encoder.classes_, cbar=False,
        )
        plt.title(f"Confusion Matrix - {display_names[mode]}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        slug = re.sub(r"[^a-z0-9]+", "_", mode.lower()).strip("_")
        plt.savefig(images_dir / f"confusion_matrix_gemini_{slug}.png", dpi=200)
        plt.close()

    # --- Save results JSON ---
    output_path = Path(output_root) / "stage2_llm_baseline_results.json"
    ensure_parent_dir(output_path)
    save_json(report, output_path)

    print(f"\nResultados salvos em: {output_path}")
    print(f"Tabela LaTeX salva em: {table_path}")
    print(f"Imagens salvas em: {images_dir}")
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 - LLM baseline (zero-shot + few-shot) for penal-merit classification"
    )
    parser.add_argument("--input", required=True, help="Path to dataset_normalized.csv")
    parser.add_argument("--prompt-file", required=True, help="Path to prompt_classification_merit.txt")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-labels", default="condenação,extinto,absolvição")
    parser.add_argument("--n-few-shot", type=int, default=3, help="Examples per class for few-shot")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between API requests (seconds)")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
    run_stage2_llm_baseline(
        input_csv=args.input,
        prompt_file=args.prompt_file,
        output_root=args.output_root,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        k_folds=args.k_folds,
        seed=args.seed,
        target_labels=target_labels,
        n_few_shot_per_class=args.n_few_shot,
        sleep_between_requests=args.sleep,
        api_key_env=args.api_key_env,
    )


if __name__ == "__main__":
    main()
