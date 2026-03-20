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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compara resultados da stage2-embeddings entre dois encoders")
    parser.add_argument("--result-a", required=True)
    parser.add_argument("--result-b", required=True)
    parser.add_argument("--output-tex", default="output/tables/table_stage2_encoder_model_comparison.tex")
    parser.add_argument("--output-png", default="output/images/figure_stage2_encoder_model_comparison_f1.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result_a = Path(args.result_a)
    result_b = Path(args.result_b)
    output_tex = Path(args.output_tex)
    output_png = Path(args.output_png)

    df, model_a_name, model_b_name = build_comparison_df(result_a, result_b)
    export_latex_table(df, model_a_name, model_b_name, output_tex)
    export_comparison_figure(df, model_a_name, model_b_name, output_png)

    print(f"Table generated: {output_tex}")
    print(f"Figure generated: {output_png}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
