"""Stage 2 – Explainability: SHAP-based interpretation for merit classifiers.

Generates embedding-level and token-level SHAP explanations for
SVM (Linear) and XGBoost classifiers trained on BERT embeddings.

Outputs:
  - Embedding-level SHAP summary bar plots (per model)
  - Side-by-side embedding dimension comparison figure
  - Token-level attribution heatmaps for selected samples
  - Force plots for selected samples
  - Structured JSON with full explainability results
  - LaTeX table with top tokens per sample
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import shap
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .io_utils import ensure_parent_dir, read_csv_smart, save_json
from .stage2_embeddings import load_dataset
from .stage2_embeddings import strip_punctuation_from_text, strip_stopwords_from_text, strip_numbers_from_text

try:
    import xgboost as xgb
except Exception:
    xgb = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _get_target_models(seed: int) -> dict:
    """Return only SVM (Linear) and XGBoost for explainability analysis."""
    models: dict[str, Any] = {
        "SVM (Linear)": SVC(
            kernel="linear",
            random_state=seed,
            class_weight="balanced",
            probability=True,  # needed for SHAP KernelExplainer
        ),
    }
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        print("[Aviso] XGBoost indisponível. Apenas SVM será analisado.")
    return models


def _select_samples(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
    n_correct_per_class: int = 1,
    n_misclassified: int = 2,
) -> list[dict]:
    """Select representative + misclassified samples for explanation."""
    samples: list[dict] = []

    # 1. One highest-confidence correct prediction per class
    for cls_idx, cls_name in enumerate(classes):
        mask = (y_true == cls_idx) & (y_pred == cls_idx)
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        confidences = proba[idxs, cls_idx]
        best = idxs[np.argmax(confidences)]
        samples.append({
            "index": int(best),
            "type": "correct",
            "true_label": cls_name,
            "pred_label": cls_name,
            "confidence": float(proba[best, cls_idx]),
            "text_preview": texts[best][:200],
        })

    # 2. Top-N misclassified samples (highest confidence in wrong class)
    mis_mask = y_true != y_pred
    if mis_mask.any():
        mis_idxs = np.where(mis_mask)[0]
        mis_conf = np.array([proba[i, y_pred[i]] for i in mis_idxs])
        top_mis = mis_idxs[np.argsort(mis_conf)[::-1][:n_misclassified]]
        for idx in top_mis:
            samples.append({
                "index": int(idx),
                "type": "misclassified",
                "true_label": classes[y_true[idx]],
                "pred_label": classes[y_pred[idx]],
                "confidence": float(proba[idx, y_pred[idx]]),
                "text_preview": texts[idx][:200],
            })

    return samples


# ---------------------------------------------------------------------------
# Embedding-level SHAP
# ---------------------------------------------------------------------------

def _compute_embedding_shap(
    model_name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    num_background: int,
) -> np.ndarray:
    """Compute SHAP values at the embedding-dimension level."""
    if "svm" in model_name.lower():
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif "xgboost" in model_name.lower():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        bg = shap.kmeans(X_train, min(num_background, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_values = explainer.shap_values(X_test, nsamples=200)
    return shap_values


def _plot_embedding_summary(
    shap_values: np.ndarray | list,
    X_test: np.ndarray,
    model_name: str,
    classes: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Generate a bar plot of top embedding dimensions by mean |SHAP|."""
    # Normalise shap_values to a 2-D array (n_samples, n_features) of
    # absolute importances aggregated across classes when necessary.
    sv_arr = np.asarray(shap_values)
    if sv_arr.ndim == 3:
        # shape (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
        # either way, take mean of abs across the class axis
        if sv_arr.shape[0] == len(classes):
            agg = np.mean(np.abs(sv_arr), axis=0)            # -> (n_samples, n_features)
        else:
            agg = np.mean(np.abs(sv_arr), axis=-1)           # -> (n_samples, n_features)
    elif isinstance(shap_values, list):
        agg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        agg = np.abs(sv_arr)

    # Ensure 2-D before collapsing across samples
    if agg.ndim != 2:
        print(f"  [Aviso] SHAP shape inesperado: {agg.shape}. Tentando achatar.")
        agg = agg.reshape(X_test.shape[0], -1)

    mean_importance = agg.mean(axis=0)
    top_dims = np.argsort(mean_importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_n),
        mean_importance[top_dims][::-1],
        color="#1f77b4",
        edgecolor="white",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"dim_{d}" for d in top_dims[::-1]], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_n} Embedding Dimensions – {model_name}")
    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _plot_embedding_comparison(
    all_shap: dict[str, np.ndarray | list],
    output_path: Path,
    top_n: int = 15,
) -> None:
    """Side-by-side bar chart comparing top embedding dimensions across models."""
    model_names = list(all_shap.keys())
    if len(model_names) < 2:
        return

    importances = {}
    for name, sv in all_shap.items():
        sv_arr = np.asarray(sv)
        if sv_arr.ndim == 3:
            agg = np.mean(np.abs(sv_arr), axis=0 if sv_arr.shape[0] != sv_arr.shape[-1] else -1)
        elif isinstance(sv, list):
            agg = np.mean([np.abs(s) for s in sv], axis=0)
        else:
            agg = np.abs(sv_arr)
        if agg.ndim > 1:
            agg = agg.mean(axis=0)
        importances[name] = agg

    # Union of top dims from both models
    all_top = set()
    for imp in importances.values():
        all_top.update(np.argsort(imp)[::-1][:top_n])
    dims = sorted(all_top)[:top_n * 2]  # cap
    dims_labels = [f"dim_{d}" for d in dims]

    fig, axes = plt.subplots(1, len(model_names), figsize=(7 * len(model_names), 6), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e"]
    for ax, (name, imp), color in zip(axes, importances.items(), colors):
        vals = imp[dims]
        order = np.argsort(vals)
        ax.barh(range(len(dims)), vals[order], color=color, edgecolor="white")
        ax.set_yticks(range(len(dims)))
        ax.set_yticklabels([dims_labels[i] for i in order], fontsize=7)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(name)

    plt.suptitle("Embedding-Level Feature Importance Comparison", fontsize=13)
    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Force plots (embedding-level, per sample)
# ---------------------------------------------------------------------------

def _plot_force(
    shap_values: np.ndarray | list,
    expected_value: Any,
    X_sample: np.ndarray,
    sample_info: dict,
    model_name: str,
    output_path: Path,
) -> None:
    """Save a SHAP force plot for a single sample."""
    # For multi-class, use the predicted class
    pred_label = sample_info["pred_label"]
    # Find class index
    if isinstance(shap_values, list):
        cls_idx = sample_info.get("pred_class_idx", 0)
        sv = shap_values[cls_idx]
        ev = expected_value[cls_idx] if hasattr(expected_value, "__getitem__") else expected_value
    else:
        sv = shap_values
        ev = expected_value

    ensure_parent_dir(output_path)
    try:
        # SHAP >= 0.40 uses shap.plots.force; fall back to legacy API
        force_fn = getattr(shap.plots, "force", None) or shap.force_plot
        fig = force_fn(
            ev,
            sv,
            X_sample,
            matplotlib=True,
            show=False,
        )
        plt.title(f"{model_name} | True: {sample_info['true_label']} | Pred: {pred_label}", fontsize=9)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {output_path}")
    except Exception as exc:
        print(f"  [Aviso] Force plot falhou para {output_path.name}: {exc}")


# ---------------------------------------------------------------------------
# Token-level SHAP via perturbation
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> list[str]:
    """Split text into word-level tokens (whitespace + punctuation aware)."""
    return re.findall(r"\S+", text)


def _build_token_classifier(
    embedder: SentenceTransformer,
    scaler: StandardScaler,
    model: Any,
    tokens: list[str],
) -> Callable:
    """Build a masking function: binary mask over tokens → predict_proba."""
    def predict_fn(masks: np.ndarray) -> np.ndarray:
        results = []
        for mask in masks:
            masked_text = " ".join(
                tok if m == 1 else "[UNK]"
                for tok, m in zip(tokens, mask)
            )
            emb = embedder.encode([masked_text], convert_to_numpy=True)
            emb_scaled = scaler.transform(emb)
            prob = model.predict_proba(emb_scaled)
            results.append(prob[0])
        return np.array(results)
    return predict_fn


def _compute_token_shap(
    text: str,
    embedder: SentenceTransformer,
    scaler: StandardScaler,
    model: Any,
    num_background: int,
    max_tokens: int = 256,
) -> tuple[list[str], np.ndarray]:
    """Compute token-level SHAP values via KernelExplainer with masking."""
    tokens = _simple_tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    predict_fn = _build_token_classifier(embedder, scaler, model, tokens)

    # Background: all tokens masked (zeros) — represents the "no information" baseline.
    # Sample: all tokens present (ones) — the actual document.
    # KernelExplainer will perturb subsets of features (tokens) between
    # these two states to measure each token's marginal contribution.
    background = np.zeros((1, len(tokens)))
    sample = np.ones((1, len(tokens)))

    explainer = shap.KernelExplainer(predict_fn, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(
            sample,
            nsamples=min(2 * len(tokens) + 100, 500),
            silent=True,
        )

    return tokens, shap_values


def _plot_token_heatmap(
    tokens: list[str],
    shap_values: np.ndarray | list,
    sample_info: dict,
    model_name: str,
    classes: list[str],
    output_path: Path,
    top_n: int = 30,
) -> None:
    """Horizontal bar chart of top-N tokens by |SHAP| for predicted class."""
    pred_label = sample_info["pred_label"]
    cls_idx = classes.index(pred_label) if pred_label in classes else 0

    if isinstance(shap_values, list):
        sv = np.array(shap_values[cls_idx]).flatten()
    else:
        sv = np.array(shap_values).flatten()

    if len(sv) != len(tokens):
        min_len = min(len(sv), len(tokens))
        sv = sv[:min_len]
        tokens = tokens[:min_len]

    abs_sv = np.abs(sv)
    top_idx = np.argsort(abs_sv)[::-1][:top_n]
    top_idx = top_idx[::-1]  # reverse for bottom-to-top plotting

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * top_n)))
    colors = ["#d62728" if sv[i] < 0 else "#2ca02c" for i in top_idx]
    ax.barh(range(len(top_idx)), sv[top_idx], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_idx)))

    # Truncate long tokens for readability
    labels = [tokens[i][:40] for i in top_idx]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("SHAP value (towards predicted class)")
    ax.set_title(
        f"Token Attribution – {model_name}\n"
        f"True: {sample_info['true_label']} | Pred: {pred_label} "
        f"({sample_info['type']})",
        fontsize=10,
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _generate_latex_table(
    all_token_results: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """Generate a LaTeX table with top-5 tokens per sample for each model."""
    rows: list[str] = []

    rows.append(r"\begin{table}[htbp]")
    rows.append(r"\centering")
    rows.append(r"\caption{Top-5 most influential tokens per sample identified by SHAP token-level attribution.}")
    rows.append(r"\label{tab:explainability}")
    rows.append(r"\resizebox{\textwidth}{!}{%")
    rows.append(r"\begin{tabular}{llllp{5cm}r}")
    rows.append(r"\toprule")
    rows.append(r"Model & Sample & Type & True$\rightarrow$Pred & Top-5 Tokens & SHAP \\")
    rows.append(r"\midrule")

    for model_name, samples in all_token_results.items():
        for i, s in enumerate(samples):
            top_tokens = s.get("top_tokens", [])[:5]
            for j, tok_info in enumerate(top_tokens):
                token_escaped = tok_info["token"].replace("_", r"\_").replace("&", r"\&")[:30]
                shap_val = f"{tok_info['shap_value']:+.4f}"
                if j == 0:
                    model_col = model_name if i == 0 else ""
                    sample_col = f"S{s['sample_index']}"
                    type_col = s["type"]
                    label_col = f"{s['true_label']}$\\rightarrow${s['pred_label']}"
                else:
                    model_col = ""
                    sample_col = ""
                    type_col = ""
                    label_col = ""
                rows.append(
                    f"{model_col} & {sample_col} & {type_col} & {label_col} "
                    f"& \\texttt{{{token_escaped}}} & {shap_val} \\\\"
                )
            if i < len(samples) - 1:
                rows.append(r"\cmidrule(lr){2-6}")
        rows.append(r"\midrule")

    # Remove last \midrule and replace with \bottomrule
    if rows[-1] == r"\midrule":
        rows[-1] = r"\bottomrule"

    rows.append(r"\end{tabular}}")
    rows.append(r"\end{table}")

    ensure_parent_dir(output_path)
    output_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Cross-model agreement
# ---------------------------------------------------------------------------

def _compute_cross_model_agreement(
    token_results: dict[str, list[dict]],
) -> list[dict]:
    """Compute Spearman rank correlation of token SHAP values between models."""
    from scipy.stats import spearmanr

    model_names = list(token_results.keys())
    if len(model_names) < 2:
        return []

    agreements = []
    m1, m2 = model_names[0], model_names[1]
    samples_1 = {s["sample_index"]: s for s in token_results[m1]}
    samples_2 = {s["sample_index"]: s for s in token_results[m2]}

    common = set(samples_1.keys()) & set(samples_2.keys())
    for idx in sorted(common):
        s1 = samples_1[idx]
        s2 = samples_2[idx]
        # Align by token list (should be identical)
        sv1 = s1.get("all_shap_values", [])
        sv2 = s2.get("all_shap_values", [])
        min_len = min(len(sv1), len(sv2))
        if min_len < 3:
            continue
        corr, pval = spearmanr(sv1[:min_len], sv2[:min_len])
        agreements.append({
            "sample_index": idx,
            "true_label": s1["true_label"],
            "type": s1["type"],
            "spearman_rho": float(corr) if not np.isnan(corr) else 0.0,
            "p_value": float(pval) if not np.isnan(pval) else 1.0,
        })

    return agreements


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_stage2_explainability(
    input_csv: str,
    output_root: str = "output",
    results_json: str | None = None,
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    model_name: str = "dominguesm/legal-bert-base-cased-ptbr",
    seed: int = 42,
    target_labels: tuple[str, ...] = ("condenação", "extinto", "absolvição"),
    num_background: int = 20,
    max_tokens: int = 256,
    strip_punctuation: bool = False,
    strip_stopwords: bool = False,
    strip_numbers: bool = False,
) -> dict:
    """Run SHAP explainability analysis on Stage 2 classifiers.

    Parameters
    ----------
    input_csv : str
        Path to the normalized dataset (same as stage2_embeddings).
    output_root : str
        Root directory for outputs.
    results_json : str, optional
        Path to stage2_embeddings_results.json (informational only).
    text_column, label_column : str
        Column names in the dataset.
    model_name : str
        SentenceTransformer model for embeddings.
    seed : int
        Random seed for reproducibility.
    target_labels : tuple
        Classes to retain.
    num_background : int
        Number of background samples for KernelExplainer.
    max_tokens : int
        Maximum tokens per document for token-level SHAP.

    Returns
    -------
    dict
        Full explainability results.
    """
    np.random.seed(seed)
    explainability_dir = Path(output_root) / "explainability"
    explainability_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & prepare data (same as stage2_embeddings)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Stage 2 – Explainability (SHAP)")
    print("=" * 60)

    df = load_dataset(input_csv, text_column=text_column, label_column=label_column)
    if target_labels:
        target_set = {x.strip().lower() for x in target_labels}
        df = df[df["label"].isin(target_set)].copy()
    print(f"Documentos após filtragem: {len(df)}")

    if strip_punctuation:
        print("[Ablation] Removendo pontuação dos textos...")
        df["text"] = df["text"].apply(strip_punctuation_from_text)
        df = df[df["text"].str.len() > 10]

    if strip_stopwords:
        print("[Ablation] Removendo stopwords dos textos...")
        df["text"] = df["text"].apply(strip_stopwords_from_text)
        df = df[df["text"].str.len() > 10]

    if strip_numbers:
        print("[Ablation] Removendo tokens numéricos dos textos...")
        df["text"] = df["text"].apply(strip_numbers_from_text)
        df = df[df["text"].str.len() > 10]

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])
    classes = encoder.classes_.tolist()
    texts = df["text"].tolist()

    # ------------------------------------------------------------------
    # 2. Embed & scale
    # ------------------------------------------------------------------
    print(f"\nGerando embeddings com {model_name}...")
    embedder = SentenceTransformer(model_name)
    X_all = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    y_all = df["label_encoded"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # ------------------------------------------------------------------
    # 3. Cross-validated predictions (to find misclassified samples)
    # ------------------------------------------------------------------
    print("\nGerando predições via cross-validation para seleção de amostras...")
    first_model_name = list(_get_target_models(seed).keys())[0]
    cv_preds = np.full(len(y_all), -1, dtype=int)
    n_classes = len(classes)
    cv_proba = np.zeros((len(y_all), n_classes))

    min_class = int(pd.Series(y_all).value_counts().min())
    cv_k = min(3, min_class)
    skf = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=seed)

    for train_idx, test_idx in skf.split(X_scaled, y_all):
        cv_model = _get_target_models(seed)[first_model_name]
        cv_scaler = StandardScaler()
        X_tr = cv_scaler.fit_transform(X_all[train_idx])
        X_te = cv_scaler.transform(X_all[test_idx])
        cv_model.fit(X_tr, y_all[train_idx])
        cv_preds[test_idx] = cv_model.predict(X_te)
        cv_proba[test_idx] = cv_model.predict_proba(X_te)

    cv_acc = accuracy_score(y_all, cv_preds)
    n_mis = int((cv_preds != y_all).sum())
    print(f"  CV accuracy ({first_model_name}): {cv_acc:.4f}  ({n_mis} misclassified)")

    # ------------------------------------------------------------------
    # 4. Train final models on full data (for SHAP analysis)
    # ------------------------------------------------------------------
    models = _get_target_models(seed)
    trained: dict[str, Any] = {}

    for name, model in models.items():
        print(f"\nTreinando {name} no dataset completo...")
        model.fit(X_scaled, y_all)
        trained[name] = model

    # ------------------------------------------------------------------
    # 5. Select 5 samples (using CV predictions to find real errors)
    # ------------------------------------------------------------------
    print(f"\nSelecionando amostras com base em CV de {first_model_name}...")
    samples = _select_samples(
        texts=texts,
        y_true=y_all,
        y_pred=cv_preds,
        proba=cv_proba,
        classes=classes,
        n_correct_per_class=1,
        n_misclassified=2,
    )
    print(f"  {len(samples)} amostras selecionadas:")
    for s in samples:
        print(f"    [{s['type']:>14s}] true={s['true_label']:<14s} pred={s['pred_label']:<14s} conf={s['confidence']:.3f}")

    # ------------------------------------------------------------------
    # 5. Embedding-level SHAP
    # ------------------------------------------------------------------
    print("\n--- Embedding-Level SHAP ---")
    all_embedding_shap: dict[str, Any] = {}
    embedding_explainers: dict[str, Any] = {}

    for name, model in trained.items():
        print(f"\n  Computing SHAP for {name}...")
        sv = _compute_embedding_shap(name, model, X_scaled, X_scaled, num_background)
        all_embedding_shap[name] = sv

        # Store explainer expected_value for force plots
        if "svm" in name.lower():
            exp = shap.LinearExplainer(model, X_scaled)
            embedding_explainers[name] = exp
        elif "xgboost" in name.lower():
            exp = shap.TreeExplainer(model)
            embedding_explainers[name] = exp

        slug = _slug(name)
        _plot_embedding_summary(
            sv, X_scaled, name, classes,
            explainability_dir / f"shap_embedding_summary_{slug}.png",
        )

    # Side-by-side comparison
    _plot_embedding_comparison(
        all_embedding_shap,
        explainability_dir / "shap_embedding_comparison.png",
    )

    # ------------------------------------------------------------------
    # 6. Force plots for selected samples (embedding-level)
    # ------------------------------------------------------------------
    print("\n--- Force Plots ---")
    for name, model in trained.items():
        slug = _slug(name)
        exp = embedding_explainers.get(name)
        if exp is None:
            continue

        sv_full = all_embedding_shap[name]
        for si, s in enumerate(samples):
            idx = s["index"]
            # Extract per-sample SHAP values
            if isinstance(sv_full, list):
                cls_idx = classes.index(s["pred_label"]) if s["pred_label"] in classes else 0
                sv_sample = sv_full[cls_idx][idx]
                ev = exp.expected_value[cls_idx] if hasattr(exp.expected_value, "__getitem__") else exp.expected_value
            else:
                sv_sample = sv_full[idx]
                ev = exp.expected_value

            sample_info = {**s, "pred_class_idx": classes.index(s["pred_label"]) if s["pred_label"] in classes else 0}
            _plot_force(
                sv_sample, ev, X_scaled[idx],
                sample_info, name,
                explainability_dir / f"shap_force_{slug}_sample_{si}.png",
            )

    # ------------------------------------------------------------------
    # 7. Token-level SHAP for selected samples
    # ------------------------------------------------------------------
    print("\n--- Token-Level SHAP (perturbation-based) ---")
    print(f"  Processing {len(samples)} samples × {len(trained)} models...")
    print("  This may take several minutes per sample.\n")

    all_token_results: dict[str, list[dict]] = {}

    for name, model in trained.items():
        slug = _slug(name)
        token_results: list[dict] = []

        for si, s in enumerate(samples):
            idx = s["index"]
            text = texts[idx]
            print(f"  [{name}] Sample {si} (idx={idx}, {s['type']})...")

            tokens, sv = _compute_token_shap(
                text, embedder, scaler, model,
                num_background=num_background,
                max_tokens=max_tokens,
            )

            # Extract SHAP values for predicted class
            pred_label = s["pred_label"]
            cls_idx = classes.index(pred_label) if pred_label in classes else 0
            if isinstance(sv, list):
                sv_cls = np.array(sv[cls_idx]).flatten()
            else:
                sv_cls = np.array(sv).flatten()

            min_len = min(len(sv_cls), len(tokens))
            sv_cls = sv_cls[:min_len]
            tokens = tokens[:min_len]

            # Top tokens
            abs_sv = np.abs(sv_cls)
            top_idx = np.argsort(abs_sv)[::-1][:10]
            top_tokens = [
                {"token": tokens[i], "shap_value": float(sv_cls[i]), "rank": int(rank + 1)}
                for rank, i in enumerate(top_idx)
            ]

            token_results.append({
                "sample_index": idx,
                "sample_order": si,
                "type": s["type"],
                "true_label": s["true_label"],
                "pred_label": s["pred_label"],
                "confidence": s["confidence"],
                "num_tokens": len(tokens),
                "top_tokens": top_tokens,
                "all_shap_values": [float(v) for v in sv_cls],
            })

            _plot_token_heatmap(
                tokens, sv_cls, s, name, classes,
                explainability_dir / f"shap_tokens_{slug}_sample_{si}.png",
            )

        all_token_results[name] = token_results

    # ------------------------------------------------------------------
    # 8. Cross-model agreement
    # ------------------------------------------------------------------
    print("\n--- Cross-Model Agreement ---")
    agreements = _compute_cross_model_agreement(all_token_results)
    if agreements:
        for a in agreements:
            print(
                f"  Sample {a['sample_index']} ({a['type']:>14s}, {a['true_label']}): "
                f"ρ = {a['spearman_rho']:+.3f}  (p = {a['p_value']:.4f})"
            )
        mean_rho = np.mean([a["spearman_rho"] for a in agreements])
        print(f"\n  Mean Spearman ρ across samples: {mean_rho:+.3f}")
    else:
        print("  Skipped (less than 2 models or insufficient data).")

    # ------------------------------------------------------------------
    # 9. LaTeX table
    # ------------------------------------------------------------------
    print("\n--- Generating LaTeX Table ---")
    table_path = Path(output_root) / "tables" / "table_explainability.tex"
    _generate_latex_table(all_token_results, table_path)

    # ------------------------------------------------------------------
    # 10. Save JSON results
    # ------------------------------------------------------------------
    results = {
        "embedding_model": model_name,
        "classes": classes,
        "models_analyzed": list(trained.keys()),
        "samples": samples,
        "token_attributions": {
            name: [
                {k: v for k, v in r.items() if k != "all_shap_values"}
                for r in recs
            ]
            for name, recs in all_token_results.items()
        },
        "cross_model_agreement": agreements,
    }

    json_path = Path(output_root) / "explainability" / "explainability_results.json"
    save_json(results, json_path)
    print(f"\nResultados salvos em: {json_path}")

    print("\n" + "=" * 60)
    print("Stage 2 – Explainability concluída.")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 2 – Explainability: SHAP analysis for SVM and XGBoost classifiers"
    )
    parser.add_argument("--input", required=True, help="Path to dataset_normalized.csv")
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--results-json", default=None, help="Path to stage2_embeddings_results.json (optional)")
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-labels", default="condenação,extinto,absolvição")
    parser.add_argument("--num-background", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--strip-punctuation", action="store_true", default=False,
                        help="Remove punctuation before embedding (ablation study)")
    parser.add_argument("--strip-stopwords", action="store_true", default=False,
                        help="Remove Portuguese stopwords before embedding (ablation study)")
    parser.add_argument("--strip-numbers", action="store_true", default=False,
                        help="Remove purely numeric tokens before embedding (ablation study)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
    run_stage2_explainability(
        input_csv=args.input,
        output_root=args.output_root,
        results_json=args.results_json,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        seed=args.seed,
        target_labels=target_labels,
        num_background=args.num_background,
        max_tokens=args.max_tokens,
        strip_punctuation=args.strip_punctuation,
        strip_stopwords=args.strip_stopwords,
        strip_numbers=args.strip_numbers,
    )


if __name__ == "__main__":
    main()
