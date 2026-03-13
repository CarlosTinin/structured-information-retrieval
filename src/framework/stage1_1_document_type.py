from __future__ import annotations

import argparse
import re

import pandas as pd

from .io_utils import ensure_parent_dir, read_csv_smart


def extract_document_type(text: str) -> str:
	if not isinstance(text, str):
		return "DESCONHECIDO"

	pattern = r"Documento id\s+\d+\s*-\s*([^\n\r]+)"
	match = re.search(pattern, text, flags=re.IGNORECASE)
	if not match:
		return "DESCONHECIDO"

	kind = match.group(1).strip().lower()
	if "sentença" in kind:
		return "Sentença"
	if "decisão" in kind:
		return "Decisão"
	if "despacho" in kind:
		return "Despacho"
	if "ata" in kind:
		return "Ata de Audiência"
	if "ato ordinatório" in kind:
		return "Ato ordinatório"
	if "acórdão" in kind:
		return "Acórdão"
	return "Outro"


def run_stage1_1(
	input_csv: str,
	output_with_type_csv: str,
	output_sentences_judgments_csv: str,
	text_column: str = "Extracted Text",
) -> pd.DataFrame:
	df = read_csv_smart(input_csv)
	if text_column not in df.columns:
		raise ValueError(f"Coluna de texto não encontrada: {text_column}")

	df["tipo_documento"] = df[text_column].apply(extract_document_type)

	ensure_parent_dir(output_with_type_csv)
	df.to_csv(output_with_type_csv, index=False)

	filtered = df[df["tipo_documento"].isin(["Sentença", "Acórdão"])].copy()
	ensure_parent_dir(output_sentences_judgments_csv)
	filtered.to_csv(output_sentences_judgments_csv, index=False)

	counts = (
		df["tipo_documento"]
		.value_counts()
		.rename_axis("tipo_documento")
		.reset_index(name="quantidade")
	)
	return counts


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Etapa 1.1 - Agrupar por tipo de documento")
	parser.add_argument("--input", required=True)
	parser.add_argument("--out-with-type", required=True)
	parser.add_argument("--out-sentencas-acordaos", required=True)
	parser.add_argument("--text-column", default="Extracted Text")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	counts = run_stage1_1(
		input_csv=args.input,
		output_with_type_csv=args.out_with_type,
		output_sentences_judgments_csv=args.out_sentencas_acordaos,
		text_column=args.text_column,
	)
	print("\nContagem por tipo:")
	print(counts.to_string(index=False))


__all__ = ["run_stage1_1", "build_arg_parser", "main"]


if __name__ == "__main__":
	main()
