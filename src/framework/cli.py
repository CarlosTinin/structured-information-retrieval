from __future__ import annotations

import argparse

from .stage1_1_document_type import run_stage1
from .stage1_2_preprocessing import run_stage2
from .stage3_embeddings import run_stage3_embeddings
from .stage3_finetune import FineTuneConfig, run_stage3_finetune
from .stage4_ner import run_stage4_ner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI da pipeline jurídica")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("stage1_1", help="Macro-etapa 1.1: classificação por tipo de documento")
    p1.add_argument("--input", required=True)
    p1.add_argument("--out-with-type", required=True)
    p1.add_argument("--out-sentencas-acordaos", required=True)
    p1.add_argument("--text-column", default="Extracted Text")

    p2 = sub.add_parser("stage1_2", help="Macro-etapa 1.2: pré-processamento textual")
    p2.add_argument("--input", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--text-column", default="Extracted Text")
    p2.add_argument("--label-column", default="decisao")
    p2.add_argument("--normalized-column", default="texto_normalizado")

    # Aliases legados para compatibilidade
    p1_legacy = sub.add_parser("stage1", help="[LEGADO] Alias para stage1_1")
    p1_legacy.add_argument("--input", required=True)
    p1_legacy.add_argument("--out-with-type", required=True)
    p1_legacy.add_argument("--out-sentencas-acordaos", required=True)
    p1_legacy.add_argument("--text-column", default="Extracted Text")

    p2_legacy = sub.add_parser("stage2", help="[LEGADO] Alias para stage1_2")
    p2_legacy.add_argument("--input", required=True)
    p2_legacy.add_argument("--output", required=True)
    p2_legacy.add_argument("--text-column", default="Extracted Text")
    p2_legacy.add_argument("--label-column", default="decisao")
    p2_legacy.add_argument("--normalized-column", default="texto_normalizado")

    p31 = sub.add_parser("stage3-finetune", help="Fine-tuning BERT")
    p31.add_argument("--input", required=True)
    p31.add_argument("--output-dir", required=True)
    p31.add_argument("--text-column", default="texto_normalizado")
    p31.add_argument("--label-column", default="decisao")
    p31.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    p31.add_argument("--epochs", type=int, default=8)
    p31.add_argument("--batch-size", type=int, default=8)
    p31.add_argument("--k-folds", type=int, default=3)

    p32 = sub.add_parser("stage3-embeddings", help="BERT embeddings + modelos clássicos")
    p32.add_argument("--input", required=True)
    p32.add_argument("--output-dir", required=True)
    p32.add_argument("--text-column", default="texto_normalizado")
    p32.add_argument("--label-column", default="decisao")
    p32.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    p32.add_argument("--k-folds", type=int, default=3)

    p4 = sub.add_parser("stage4", help="Extração NER")
    p4.add_argument("--input-json", required=True)
    p4.add_argument("--output-json", required=True)
    p4.add_argument("--output-csv", required=True)
    p4.add_argument("--model-name", default="dominguesm/legal-bert-ner-base-cased-ptbr")
    p4.add_argument("--sentence-key", default="sentenca")
    p4.add_argument("--label-key", default="label")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"stage1_1", "stage1"}:
        counts = run_stage1(args.input, args.out_with_type, args.out_sentencas_acordaos, args.text_column)
        print(counts.to_string(index=False))
        return

    if args.command in {"stage1_2", "stage2"}:
        print(run_stage2(args.input, args.output, args.text_column, args.label_column, args.normalized_column))
        return

    if args.command == "stage3-finetune":
        cfg = FineTuneConfig(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            k_folds=args.k_folds,
        )
        print(run_stage3_finetune(args.input, args.output_dir, args.text_column, args.label_column, cfg))
        return

    if args.command == "stage3-embeddings":
        print(run_stage3_embeddings(args.input, args.output_dir, args.text_column, args.label_column, args.model_name, args.k_folds))
        return

    if args.command == "stage4":
        print(run_stage4_ner(args.input_json, args.output_json, args.output_csv, args.model_name, args.sentence_key, args.label_key))
        return


if __name__ == "__main__":
    main()
