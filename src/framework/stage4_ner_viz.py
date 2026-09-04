"""Stage 4 – NER visualisations.

Generates publication-quality figures and tables from NER extraction results:

* **Composite heatmap** (``fig2_ner_heatmap``): section × entity-type count
  matrix with marginal row/column totals, replacing both a standalone bar
  chart and a separate heatmap with a single dense figure.
* **Single-document LaTeX table** (appendix material): detailed extraction
  results for one representative document.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io_utils import ensure_parent_dir


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

# Logical ordering for document sections (top → bottom).
_SECTION_ORDER = [
    "RELATORIO",
    "DOS_FATOS",
    "FUNDAMENTACAO",
    "DOSIMETRIA",
    "DISPOSITIVO",
    "OUTROS",
]

# Logical ordering for entity types (left → right).
_ENTITY_ORDER = [
    "PESSOA",
    "ORGANIZACAO",
    "LOCAL",
    "TEMPO",
    "LEGISLACAO",
    "JURISPRUDENCIA",
]


def _load_by_section(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_count_matrix(data: list[dict]) -> pd.DataFrame:
    """Build a *section × entity_type* count matrix from by-section JSON."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for doc in data:
        for sec in doc["sections"]:
            section = sec["section"]
            for etype, ents in sec["entities_by_type"].items():
                counts[section][etype] += len(ents)

    # Build DataFrame with controlled ordering
    sections = [s for s in _SECTION_ORDER if s in counts]
    etypes = [e for e in _ENTITY_ORDER if any(e in counts[s] for s in sections)]

    matrix = pd.DataFrame(
        [[counts[s][e] for e in etypes] for s in sections],
        index=sections,
        columns=etypes,
    )
    return matrix


# ------------------------------------------------------------------
# Figure: composite heatmap with marginal totals
# ------------------------------------------------------------------


def plot_ner_heatmap(
    data: list[dict],
    output_dir: str,
    *,
    figname: str = "fig2_ner_heatmap",
    cmap: str = "YlOrRd",
    dpi: int = 600,
) -> str:
    """Create a section × entity-type heatmap with marginal totals.

    The right-most column shows the row total (entities per section) and
    the bottom row shows the column total (entities per type), effectively
    embedding the entity-type distribution that would otherwise require a
    separate bar chart.

    Designed for two-column journal figure width (~17.8 cm / 7 in).
    Font sizes increased for print legibility.

    Returns the path of the saved PNG.
    """
    matrix = _build_count_matrix(data)

    # Append marginal totals
    matrix["Total"] = matrix.sum(axis=1)
    totals_row = matrix.sum(axis=0)
    totals_row.name = "Total"
    matrix = pd.concat([matrix, totals_row.to_frame().T])

    n_rows, n_cols = matrix.shape

    # --- Publication-quality figure layout (two-column width, slightly compact) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Mask for the marginal cells (last row and last column) so the main
    # body uses the colour-map while totals are visually distinct.
    mask_main = np.zeros_like(matrix.values, dtype=bool)
    mask_main[-1, :] = True  # total row
    mask_main[:, -1] = True  # total column

    mask_margin = ~mask_main

    # Draw main body
    sns.heatmap(
        matrix,
        ax=ax,
        mask=mask_main,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Entity count", "shrink": 0.75},
        annot_kws={"fontsize": 10},
    )

    # Draw marginal totals with a neutral grey background
    sns.heatmap(
        matrix,
        ax=ax,
        mask=mask_margin,
        annot=True,
        fmt="d",
        cmap=["#f0f0f0"],
        linewidths=0.6,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_title(
        "Named-Entity Distribution by Document Section",
        fontsize=12,
        pad=10,
    )

    # Update colorbar label font size
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label("Entity count", fontsize=10)

    fig.tight_layout()

    out_png = str(Path(output_dir) / "images" / f"{figname}.png")
    out_pdf = str(Path(output_dir) / "images" / f"{figname}.pdf")
    ensure_parent_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[stage4-viz] Heatmap saved → {out_png}")
    print(f"[stage4-viz] Heatmap saved → {out_pdf}")
    return out_png


# ------------------------------------------------------------------
# Table: single-document NER extraction (LaTeX, appendix material)
# ------------------------------------------------------------------


def generate_single_doc_table(
    data: list[dict],
    doc_id: int,
    output_dir: str,
    *,
    tablename: str = "table_ner_single_doc",
) -> str:
    """Generate a LaTeX table with NER results for one document.

    Columns: Section | Entity Type | Entity Text | Score.
    Returns the path of the saved ``.tex`` file.
    """
    doc = next((d for d in data if d["doc_id"] == doc_id), None)
    if doc is None:
        raise ValueError(
            f"doc_id={doc_id} not found. "
            f"Available: {sorted(d['doc_id'] for d in data)}"
        )

    rows: list[dict] = []
    for sec in doc["sections"]:
        section = sec["section"]
        for etype in _ENTITY_ORDER:
            for ent in sec["entities_by_type"].get(etype, []):
                rows.append(
                    {
                        "Section": section,
                        "Entity Type": etype,
                        "Entity": ent["text"],
                        "Score": f"{ent['score']:.4f}",
                    }
                )

    df = pd.DataFrame(rows)

    # Collapse repeated Section / Entity Type values for readability
    prev_sec = None
    prev_etype = None
    for i in range(len(df)):
        sec = df.at[i, "Section"]
        etype = df.at[i, "Entity Type"]
        if sec == prev_sec:
            df.at[i, "Section"] = ""
        else:
            prev_sec = sec
            prev_etype = None  # reset entity type grouping on section change
        if etype == prev_etype and df.at[i, "Section"] == "":
            df.at[i, "Entity Type"] = ""
        else:
            prev_etype = etype

    # Escape LaTeX special characters in entity text
    def _escape(s: str) -> str:
        for ch in ("&", "%", "$", "#", "_", "{", "}"):
            s = s.replace(ch, f"\\{ch}")
        return s

    df["Entity"] = df["Entity"].apply(_escape)

    latex = df.to_latex(
        index=False,
        column_format="llp{6cm}r",
        caption=(
            f"Named entities extracted from document {doc_id} "
            f"(representative example)."
        ),
        label=f"tab:ner_doc{doc_id}",
        escape=False,
    )

    out_path = str(Path(output_dir) / "tables" / f"{tablename}.tex")
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex)

    print(f"[stage4-viz] LaTeX table saved → {out_path}")
    return out_path


# ------------------------------------------------------------------
# Figure: entity type co-occurrence by section (vertical layout)
# ------------------------------------------------------------------

# Sections to display in the co-occurrence figure
_COOCCURRENCE_SECTIONS = ["DOS_FATOS", "DISPOSITIVO", "FUNDAMENTACAO"]


def _build_cooccurrence_matrix_for_section(
    by_sentence_data: list[dict],
    section: str,
) -> pd.DataFrame:
    """Build a pairwise entity-type co-occurrence matrix for a given section.

    A co-occurrence is counted when a sentence contains both entity types.
    The diagonal counts sentences containing that entity type (regardless of
    co-occurrence with another type).
    """
    # Initialize matrix with entity types
    etypes = [e for e in _ENTITY_ORDER]
    n = len(etypes)
    matrix = np.zeros((n, n), dtype=int)
    etype_to_idx = {e: i for i, e in enumerate(etypes)}

    for doc in by_sentence_data:
        for sent in doc.get("sentences", []):
            if sent.get("section", "") != section:
                continue
            types_present = [
                t for t in sent.get("entities_by_type", {}).keys()
                if t in etype_to_idx
            ]
            # Count diagonal: sentences containing each type
            for t in types_present:
                matrix[etype_to_idx[t], etype_to_idx[t]] += 1
            # Count off-diagonal: pairwise co-occurrences
            for i_idx in range(len(types_present)):
                for j_idx in range(i_idx + 1, len(types_present)):
                    ti = etype_to_idx[types_present[i_idx]]
                    tj = etype_to_idx[types_present[j_idx]]
                    matrix[ti, tj] += 1
                    matrix[tj, ti] += 1

    return pd.DataFrame(matrix, index=etypes, columns=etypes)


def plot_cooccurrence_vertical(
    by_sentence_data: list[dict],
    output_dir: str,
    *,
    figname: str = "fig3_cooccurrence_heatmap",
    sections: list[str] | None = None,
    dpi: int = 600,
) -> str:
    """Create vertically stacked co-occurrence heatmaps (one per section).

    Designed for single-column journal figure (~8.5 cm / 3.5 in width) or
    two-column width (~17.8 cm / 7 in) depending on available space.
    Fonts are increased for print legibility; layout is vertical (one subplot
    above the other) rather than horizontal side-by-side.

    Returns the path of the saved PNG.
    """
    if sections is None:
        sections = _COOCCURRENCE_SECTIONS

    n_sections = len(sections)

    # Publication-quality settings
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Vertical layout: one subplot per section, stacked — larger for readability
    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(6.5, 5.2 * n_sections),
    )
    if n_sections == 1:
        axes = [axes]

    # Consistent color scale across subplots
    all_matrices = []
    for section in sections:
        mat = _build_cooccurrence_matrix_for_section(by_sentence_data, section)
        all_matrices.append(mat)

    # Use the global max (excluding diagonal) for consistent scale
    global_max = max(
        mat.values[np.triu_indices_from(mat.values, k=1)].max()
        for mat in all_matrices
        if mat.values[np.triu_indices_from(mat.values, k=1)].size > 0
    )

    for ax, section, mat in zip(axes, sections, all_matrices):
        # Count sentences with co-occurrence for subtitle
        n_sentences = 0
        for doc in by_sentence_data:
            for sent in doc.get("sentences", []):
                if sent.get("section", "") == section:
                    types_in_sent = [
                        t for t in sent.get("entities_by_type", {}).keys()
                        if t in {e for e in _ENTITY_ORDER}
                    ]
                    if len(types_in_sent) >= 2:
                        n_sentences += 1

        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 13},
            vmin=0,
            vmax=max(global_max, 1),
        )

        ax.set_title(
            f"{section}",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        # Update colorbar font
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Entity Type Co-occurrence by Section",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = str(Path(output_dir) / "images" / f"{figname}.png")
    out_pdf = str(Path(output_dir) / "images" / f"{figname}.pdf")
    ensure_parent_dir(out_png)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"[stage4-viz] Co-occurrence figure saved → {out_png}")
    print(f"[stage4-viz] Co-occurrence figure saved → {out_pdf}")
    return out_png


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


def run_stage4_viz(
    input_by_section: str,
    output_root: str = "output",
    doc_id: int = 0,
    input_by_sentence: str | None = None,
) -> dict:
    """Run all stage-4 visualisations and return a summary dict."""
    data = _load_by_section(input_by_section)

    heatmap_path = plot_ner_heatmap(data, output_root)
    table_path = generate_single_doc_table(data, doc_id, output_root)

    cooccurrence_path = None
    if input_by_sentence:
        by_sentence_data = _load_by_section(input_by_sentence)
        cooccurrence_path = plot_cooccurrence_vertical(by_sentence_data, output_root)

    return {
        "heatmap": heatmap_path,
        "cooccurrence": cooccurrence_path,
        "table": table_path,
        "num_documents": len(data),
        "doc_id_for_table": doc_id,
    }
