from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter

from tqdm import tqdm

from .io_utils import ensure_parent_dir, read_csv_smart


def normalize_typography(text: str) -> str:
	text = text.lower()
	text = unicodedata.normalize("NFKC", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def normalize_typography_for_ner(text: str) -> str:
	"""Normalização leve para NER/segmentação (preserva mais informação)."""
	text = unicodedata.normalize("NFKC", text)
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = re.sub(r"[ \t]+", " ", text)
	text = re.sub(r"\n{3,}", "\n\n", text)
	return text.strip()


def remove_duplicate_footer(text: str) -> tuple[str, int]:
	"""Remove rodapés repetidos do PJe presentes em várias páginas."""
	if not isinstance(text, str):
		return "", 0

	patterns = [
		r"(?im)^\s*Num\.\s*\d+\s*-\s*Pág\.\s*\d+\s*$",
		r"(?im)^\s*Assinado eletronicamente por:.*$",
		r"(?im)^\s*https?://\S*consultapublica/Processo/ConsultaDocumento/listView\.seam\?x=\S+\s*$",
		r"(?im)^\s*Número do documento:\s*\d+\s*$",
		r"(?im)^\s*Documento id\s+\d+\s*-\s*[^\n\r]+$",
		r"(?i)Número do documento:\s*\d+\s*Documento id\s+\d+\s*-\s*[^\n\r]+",
	]

	removed = 0
	current = text
	for pattern in patterns:
		current, n = re.subn(pattern, " ", current)
		removed += n

	current = re.sub(r"\n{3,}", "\n\n", current)
	current = re.sub(r"[ \t]{2,}", " ", current)
	return current.strip(), removed


def normalize_legal_text(text: str, min_size: int = 30) -> tuple[str, Counter]:
	counters: Counter = Counter()
	if not isinstance(text, str):
		return "", counters

	original = text
	boilerplates = [
		r"vistos[, ]*etc\.?",
		r"é o relatório",
		r"relatório dispensado",
		r"passo a decidir",
		r"ante o exposto",
		r"publique-se[, ]*registre-se[, ]*intimem-se\.?",
	]
	for pattern in boilerplates:
		updated, n = re.subn(pattern, " ", text, flags=re.IGNORECASE)
		text = updated
		counters["boilerplate"] += n

	patterns = {
		"cnj": r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}",
		"numeros_extensos": r"\b\d{6,}\b",
		"nomes_caps": r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,}(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{2,})+\b",
		"partes": r"\b(autor|réu|apelante|apelado|impetrante|impetrado)\b",
		"valores": r"r\$ ?\d+[.,\d]*",
	}

	for name, pattern in patterns.items():
		text, n = re.subn(pattern, " ", text, flags=re.IGNORECASE)
		counters[name] += n

	for date_pattern in [r"\d{2}/\d{2}/\d{4}", r"\d{1,2} de [a-zç]+ de \d{4}"]:
		text, n = re.subn(date_pattern, " ", text, flags=re.IGNORECASE)
		counters["datas"] += n

	for legal_pattern in [r"art\.? ?\d+[ºo]?", r"artigo ?\d+", r"§ ?\d+º?", r"inciso [ivxlcdm]+"]:
		text, n = re.subn(legal_pattern, " ", text, flags=re.IGNORECASE)
		counters["referencias_legais"] += n

	text = normalize_typography(text)
	if len(text) < min_size:
		return original, counters

	return text, counters


def normalize_text_for_ner(text: str) -> str:
	if not isinstance(text, str):
		return ""
	return normalize_typography_for_ner(text)


def run_stage1_2(
	input_csv: str,
	output_classification_csv: str,
	output_ner_csv: str,
	text_column: str = "Extracted Text",
	label_column: str = "decisao",
	classification_column: str = "texto_normalizado",
	ner_column: str = "texto_ner",
) -> dict:
	df = read_csv_smart(input_csv)
	for col in (text_column, label_column):
		if col not in df.columns:
			raise ValueError(f"Coluna obrigatória não encontrada: {col}")

	global_counter: Counter = Counter()
	classification_texts: list[str] = []
	ner_texts: list[str] = []
	footer_rows_affected = 0

	for raw in tqdm(df[text_column].tolist(), desc="Normalizando"):
		without_footer, removed_footer = remove_duplicate_footer(raw)
		if removed_footer > 0:
			footer_rows_affected += 1
		global_counter["footer_duplicado"] += removed_footer

		normalized_classification, counters = normalize_legal_text(without_footer)
		normalized_ner = normalize_text_for_ner(without_footer)

		classification_texts.append(normalized_classification)
		ner_texts.append(normalized_ner)
		global_counter.update(counters)

	df[classification_column] = classification_texts
	df[ner_column] = ner_texts

	export_classification = df[[text_column, classification_column, label_column]].copy()
	ensure_parent_dir(output_classification_csv)
	export_classification.to_csv(output_classification_csv, index=False)

	export_ner = df[[text_column, ner_column, label_column]].copy()
	ensure_parent_dir(output_ner_csv)
	export_ner.to_csv(output_ner_csv, index=False)

	avg_before = float(df[text_column].astype(str).str.len().mean())
	avg_after_classification = float(df[classification_column].astype(str).str.len().mean())
	avg_after_ner = float(df[ner_column].astype(str).str.len().mean())
	changed_docs_classification = int((df[text_column].astype(str) != df[classification_column].astype(str)).sum())
	changed_docs_ner = int((df[text_column].astype(str) != df[ner_column].astype(str)).sum())

	return {
		"total_docs": int(len(df)),
		"changed_docs_classification": changed_docs_classification,
		"changed_docs_ner": changed_docs_ner,
		"footer_rows_affected": footer_rows_affected,
		"avg_len_before": avg_before,
		"avg_len_after_classification": avg_after_classification,
		"avg_len_after_ner": avg_after_ner,
		"output_classification_csv": output_classification_csv,
		"output_ner_csv": output_ner_csv,
		"transform_counts": dict(global_counter),
	}


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Etapa 1.2 - Pré-processamento para classificadores")
	parser.add_argument("--input", default="files/output/dataset_filtered_by_type.csv")
	parser.add_argument("--output-classification", default="files/output/dataset_normalized.csv")
	parser.add_argument("--output-ner", default="files/output/dataset_normalized_for_ner.csv")
	parser.add_argument("--text-column", default="Extracted Text")
	parser.add_argument("--label-column", default="decisao")
	parser.add_argument("--classification-column", default="texto_normalizado")
	parser.add_argument("--ner-column", default="texto_ner")
	return parser


def main() -> None:
	args = build_arg_parser().parse_args()
	report = run_stage1_2(
		input_csv=args.input,
		output_classification_csv=args.output_classification,
		output_ner_csv=args.output_ner,
		text_column=args.text_column,
		label_column=args.label_column,
		classification_column=args.classification_column,
		ner_column=args.ner_column,
	)
	print("\nRelatório da etapa 1.2:")
	for k, v in report.items():
		print(f"- {k}: {v}")


__all__ = ["run_stage1_2", "build_arg_parser", "main"]


if __name__ == "__main__":
	main()
