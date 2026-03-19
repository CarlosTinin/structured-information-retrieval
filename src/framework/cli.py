from __future__ import annotations

import argparse

from .stage1_1_document_type import run_stage1_1
from .stage1_2_preprocessing import run_stage1_2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI da pipeline jurídica")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("stage1_1", help="Macro-etapa 1.1: classificação por tipo de documento")
    p1.add_argument("--input", default="files/datasets/dataset_completo.csv")
    p1.add_argument("--output", default="files/output/dataset_filtered_by_type.csv")
    p1.add_argument("--out-with-type", default=None)
    p1.add_argument("--text-column", default="Extracted Text")

    p2 = sub.add_parser("stage1_2", help="Macro-etapa 1.2: pré-processamento textual")
    p2.add_argument("--input", default="files/output/dataset_filtered_by_type.csv")
    p2.add_argument("--output-classification", default="files/output/dataset_normalized.csv")
    p2.add_argument("--output-ner", default="files/output/dataset_normalized_for_ner.csv")
    p2.add_argument("--text-column", default="Extracted Text")
    p2.add_argument("--label-column", default="decisao")
    p2.add_argument("--classification-column", default="texto_normalizado")
    p2.add_argument("--ner-column", default="texto_ner")

    p2f = sub.add_parser("stage2-finetune", help="Etapa 2 (fine-tuning BERT)")
    p2f.add_argument("--input", required=True)
    p2f.add_argument("--output-dir", required=True)
    p2f.add_argument("--text-column", default="texto_normalizado")
    p2f.add_argument("--label-column", default="decisao")
    p2f.add_argument("--model-name", default="neuralmind/bert-base-portuguese-cased")
    p2f.add_argument("--epochs", type=int, default=8)
    p2f.add_argument("--batch-size", type=int, default=8)
    p2f.add_argument("--k-folds", type=int, default=3)

    p2e = sub.add_parser("stage2-embeddings", help="Etapa 2 (embeddings BERT + modelos clássicos)")
    p2e.add_argument("--input", required=True)
    p2e.add_argument("--output-root", default="output")
    p2e.add_argument("--text-column", default="texto_normalizado")
    p2e.add_argument("--label-column", default="decisao")
    p2e.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    p2e.add_argument("--k-folds", type=int, default=3)
    p2e.add_argument("--target-labels", default="condenação,extinto,absolvição")

    p5 = sub.add_parser("stage5", help="Etapa 5 - Extração NER")
    p5.add_argument("--input-json", required=True)
    p5.add_argument("--output-json", required=True)
    p5.add_argument("--output-csv", required=True)
    p5.add_argument("--model-name", default="dominguesm/legal-bert-ner-base-cased-ptbr")
    p5.add_argument("--sentence-key", default="sentenca")
    p5.add_argument("--label-key", default="label")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stage1_1":
        counts = run_stage1_1(args.input, args.output, args.out_with_type, args.text_column)
        print(counts.to_string(index=False))
        return

    if args.command == "stage1_2":
        print(
            run_stage1_2(
                args.input,
                args.output_classification,
                args.output_ner,
                args.text_column,
                args.label_column,
                args.classification_column,
                args.ner_column,
            )
        )
        return

    if args.command == "stage2-finetune":
        from .stage2_finetune import FineTuneConfig, run_stage2_finetune

        cfg = FineTuneConfig(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            k_folds=args.k_folds,
        )
        print(run_stage2_finetune(args.input, args.output_dir, args.text_column, args.label_column, cfg))
        return

    if args.command == "stage2-embeddings":
        from .stage2_embeddings import run_stage2_embeddings

        target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
        print(
            run_stage2_embeddings(
                input_csv=args.input,
                output_root=args.output_root,
                text_column=args.text_column,
                label_column=args.label_column,
                model_name=args.model_name,
                k_folds=args.k_folds,
                target_labels=target_labels,
            )
        )
        return

    if args.command == "stage5":
        from .stage5_ner import run_stage5_ner

        print(run_stage5_ner(args.input_json, args.output_json, args.output_csv, args.model_name, args.sentence_key, args.label_key))
        return


if __name__ == "__main__":
    main()
