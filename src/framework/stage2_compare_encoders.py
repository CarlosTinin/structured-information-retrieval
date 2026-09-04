from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_comparison_df(result_a_path: Path, result_b_path: Path) -> tuple[pd.DataFrame, str, str]:
    with result_a_path.open("r", encoding="utf-8") as f:
        a = json.load(f)
    with result_b_path.open("r", encoding="utf-8") as f:
        b = json.load(f)

    model_a_name = str(a.get("embedding_model", "model_a"))
    model_b_name = str(b.get("embedding_model", "model_b"))

    models_a = a.get("models", {}) if isinstance(a, dict) else {}
    models_b = b.get("models", {}) if isinstance(b, dict) else {}
    common = sorted(set(models_a).intersection(models_b))
    if not common:
        raise RuntimeError("Nenhum classificador em comum entre os dois resultados.")

    rows = []
    for clf in common:
        f1_a = float(models_a[clf].get("mean_f1", 0.0))
        std_a = float(models_a[clf].get("std_f1", 0.0))
        f1_b = float(models_b[clf].get("mean_f1", 0.0))
        std_b = float(models_b[clf].get("std_f1", 0.0))
        rows.append(
            {
                "Classifier": clf,
                "F1_model_a": f1_a,
                "STD_model_a": std_a,
                "F1_model_b": f1_b,
                "STD_model_b": std_b,
                "Delta_model_b_minus_model_a": f1_b - f1_a,
            }
        )

    df = pd.DataFrame(rows).sort_values("F1_model_a", ascending=False).reset_index(drop=True)
    return df, model_a_name, model_b_name


def build_multi_encoder_df(result_paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """Build a comparison DataFrame for multiple (2+) encoder results.

    Returns a DataFrame with columns: Classifier, F1_<model>, STD_<model>, ...
    and a list of model names.
    """
    all_data = []
    model_names = []
    for rpath in result_paths:
        with rpath.open("r", encoding="utf-8") as f:
            d = json.load(f)
        all_data.append(d)
        model_names.append(str(d.get("embedding_model", rpath.stem)))

    # Find common classifiers across all encoder results
    all_clf_sets = [set(d.get("models", {}).keys()) for d in all_data]
    common = sorted(set.intersection(*all_clf_sets))
    if not common:
        raise RuntimeError("No classifiers in common across all encoder results.")

    rows = []
    for clf in common:
        row = {"Classifier": clf}
        for i, d in enumerate(all_data):
            m = d["models"][clf]
            row[f"F1_{i}"] = float(m.get("mean_f1", 0.0))
            row[f"STD_{i}"] = float(m.get("std_f1", 0.0))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, model_names


def export_latex_table(df: pd.DataFrame, model_a_name: str, model_b_name: str, output_tex: Path) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    tbl = df.copy()
    for col in [
        "F1_model_a",
        "STD_model_a",
        "F1_model_b",
        "STD_model_b",
        "Delta_model_b_minus_model_a",
    ]:
        tbl[col] = tbl[col].map(lambda x: f"{x:.4f}")

    tbl = tbl.rename(
        columns={
            "Classifier": "Classifier",
            "F1_model_a": f"F1 ({model_a_name})",
            "STD_model_a": f"Std ({model_a_name})",
            "F1_model_b": f"F1 ({model_b_name})",
            "STD_model_b": f"Std ({model_b_name})",
            "Delta_model_b_minus_model_a": "$\\Delta$ (model\_b - model\_a)",
        }
    )

    output_tex.write_text(tbl.to_latex(index=False, escape=False), encoding="utf-8")


def export_comparison_figure(df: pd.DataFrame, model_a_name: str, model_b_name: str, output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=300)

    x = np.arange(len(df))
    width = 0.36

    ax.bar(
        x - width / 2,
        df["F1_model_a"],
        width,
        yerr=df["STD_model_a"],
        capsize=4,
        label=model_a_name,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        df["F1_model_b"],
        width,
        yerr=df["STD_model_b"],
        capsize=4,
        label=model_b_name,
        color="#55A868",
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Classifier"], rotation=18, ha="right")
    ax.set_ylabel("Weighted F1 (mean across folds)")
    ax.set_xlabel("Classifier")
    ax.set_title("Stage 2 encoder comparison by downstream classifier")
    ax.set_ylim(0.0, float(max(df["F1_model_a"].max(), df["F1_model_b"].max()) + 0.15))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=1, frameon=False, fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def export_multi_encoder_figure(
    df: pd.DataFrame,
    model_names: list[str],
    output_png: Path,
    *,
    dpi: int = 600,
) -> None:
    """Generate a publication-quality two-column-width grouped bar chart for
    multiple encoder models.

    Designed for journal two-column figures (~17.8 cm / 7 in width).
    Increased font sizes and high DPI for print quality.
    """
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Shorten model names for legend readability
    short_names = []
    for name in model_names:
        # Extract meaningful short name from HuggingFace model paths
        if "legal-bert" in name.lower():
            short_names.append("Legal-BERT (dominguesm)")
        elif "bert-large" in name.lower():
            short_names.append("BERTimbau-large (neuralmind)")
        elif "bert-base" in name.lower() and "neuralmind" in name.lower():
            short_names.append("BERTimbau-base (neuralmind)")
        else:
            # Fallback: use last part of path
            short_names.append(name.split("/")[-1] if "/" in name else name)

    # Publication-quality settings
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    # Two-column width figure for Elsevier journals
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=dpi)

    n_models = len(model_names)
    n_classifiers = len(df)
    x = np.arange(n_classifiers)
    total_bar_width = 0.78
    width = total_bar_width / n_models

    # Colour palette (colourblind-friendly)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][:n_models]

    for i, (sname, color) in enumerate(zip(short_names, colors)):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            df[f"F1_{i}"],
            width * 0.92,
            yerr=df[f"STD_{i}"],
            capsize=3,
            label=sname,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            error_kw={"linewidth": 0.9},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Classifier"], rotation=0, ha="center", fontsize=11)
    ax.set_ylabel("Weighted F1 (mean ± std)", fontsize=12)
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_title("Stage 2 encoder comparison by downstream classifier", fontsize=13, pad=8)

    max_f1 = max(df[f"F1_{i}"].max() for i in range(n_models))
    ax.set_ylim(0.0, min(float(max_f1 + 0.15), 1.0))
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="lightgrey",
        fontsize=9.5,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Save in multiple formats for publication
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    pdf_path = output_png.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"[stage2] Multi-encoder figure saved → {output_png}")
    print(f"[stage2] Multi-encoder figure saved → {pdf_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara resultados da stage2-embeddings entre dois ou mais encoders")
    parser.add_argument("--result-a", help="Path to first encoder result JSON (legacy 2-encoder mode)")
    parser.add_argument("--result-b", help="Path to second encoder result JSON (legacy 2-encoder mode)")
    parser.add_argument(
        "--results",
        nargs="+",
        help="Paths to multiple encoder result JSONs (multi-encoder mode, preferred)",
    )
    parser.add_argument("--output-tex", default="output/tables/table_stage2_encoder_model_comparison.tex")
    parser.add_argument("--output-png", default="output/images/figure_stage2_encoder_model_comparison_f1.png")
    parser.add_argument("--dpi", type=int, default=600, help="Output figure DPI (default: 600)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_tex = Path(args.output_tex)
    output_png = Path(args.output_png)

    if args.results and len(args.results) >= 2:
        # Multi-encoder mode (new, preferred)
        result_paths = [Path(p) for p in args.results]
        df, model_names = build_multi_encoder_df(result_paths)
        export_multi_encoder_figure(df, model_names, output_png, dpi=args.dpi)
        print(f"Figure generated: {output_png}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    elif args.result_a and args.result_b:
        # Legacy 2-encoder mode
        result_a = Path(args.result_a)
        result_b = Path(args.result_b)
        df, model_a_name, model_b_name = build_comparison_df(result_a, result_b)
        export_latex_table(df, model_a_name, model_b_name, output_tex)
        export_comparison_figure(df, model_a_name, model_b_name, output_png)
        print(f"Table generated: {output_tex}")
        print(f"Figure generated: {output_png}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("Error: provide either --results (2+ paths) or --result-a and --result-b")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
