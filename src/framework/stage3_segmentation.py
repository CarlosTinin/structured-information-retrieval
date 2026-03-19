from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent_dir, read_csv_smart, save_json


def _read_env_file_key(env_path: Path, key: str) -> str:
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() != key:
                continue
            return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _resolve_api_key(key_name: str) -> str:
    api_key = os.getenv(key_name, "").strip()
    if api_key:
        return api_key

    # fallback: tenta .env no diretório atual e na raiz do projeto
    cwd_env = Path(".env")
    key = _read_env_file_key(cwd_env, key_name)
    if key:
        os.environ[key_name] = key
        return key

    project_env = Path(__file__).resolve().parents[2] / ".env"
    key = _read_env_file_key(project_env, key_name)
    if key:
        os.environ[key_name] = key
        return key

    return ""


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Extrai JSON válido da resposta do modelo (com ou sem markdown fence)."""
    cleaned = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    for candidate in (cleaned,):
        try:
            payload = json.loads(candidate)
            if isinstance(payload, list):
                return {"total_sentencas": len(payload), "dados": payload}
            if isinstance(payload, dict):
                if "dados" in payload and isinstance(payload["dados"], list):
                    if "total_sentencas" not in payload:
                        payload["total_sentencas"] = len(payload["dados"])
                    return payload
                return {"total_sentencas": 0, "dados": []}
        except json.JSONDecodeError:
            pass

    # fallback: tenta encontrar primeiro bloco JSON no texto
    block_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if block_match:
        candidate = block_match.group(1)
        payload = json.loads(candidate)
        if isinstance(payload, list):
            return {"total_sentencas": len(payload), "dados": payload}
        if isinstance(payload, dict) and "dados" in payload and isinstance(payload["dados"], list):
            if "total_sentencas" not in payload:
                payload["total_sentencas"] = len(payload["dados"])
            return payload

    raise ValueError("Resposta da LLM não contém JSON válido no formato esperado.")


def _build_prompt(base_prompt: str, text: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "==================== TEXTO PARA ANOTAÇÃO ====================\n"
        f"{text}"
    )


def run_stage3_segmentation(
    input_csv: str,
    prompt_file: str,
    output_json: str,
    model_name: str = "gemini-2.0-flash",
    text_column: str = "texto_ner",
    id_column: str = "id",
    max_docs: int | None = None,
    sleep_seconds: float = 0.0,
    api_key_env: str = "GEMINI_API_KEY",
) -> dict[str, Any]:
    api_key = _resolve_api_key(api_key_env)
    if not api_key:
        raise ValueError(f"Chave da API não encontrada. Defina a variável de ambiente {api_key_env}.")

    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "Pacote 'google-generativeai' não encontrado. Instale com: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    df = read_csv_smart(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Coluna de texto não encontrada: {text_column}")

    prompt_text = Path(prompt_file).read_text(encoding="utf-8")

    records = df.to_dict("records")
    if max_docs is not None:
        records = records[:max_docs]

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(records):
        raw_text = str(row.get(text_column, ""))
        doc_id = row.get(id_column, idx)

        if not raw_text.strip():
            errors.append({"index": idx, "doc_id": doc_id, "error": "Texto vazio"})
            continue

        prompt = _build_prompt(prompt_text, raw_text)
        try:
            response = model.generate_content(prompt)
            response_text = getattr(response, "text", "") or ""
            payload = _extract_json_payload(response_text)
            results.append(
                {
                    "index": idx,
                    "doc_id": doc_id,
                    "total_sentencas": int(payload.get("total_sentencas", len(payload.get("dados", [])))),
                    "dados": payload.get("dados", []),
                }
            )
        except Exception as exc:
            errors.append({"index": idx, "doc_id": doc_id, "error": str(exc)})

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    output_payload = {
        "model_name": model_name,
        "input_csv": input_csv,
        "text_column": text_column,
        "total_documentos": len(records),
        "processados_com_sucesso": len(results),
        "com_erro": len(errors),
        "resultados": results,
        "erros": errors,
    }

    ensure_parent_dir(output_json)
    save_json(output_payload, output_json)

    return {
        "total_documentos": len(records),
        "processados_com_sucesso": len(results),
        "com_erro": len(errors),
        "output_json": output_json,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 3 - Segmentação com LLM (Gemini)")
    parser.add_argument("--input", default="files/output/dataset_normalized_for_ner.csv")
    parser.add_argument("--prompt-file", default="src/prompts/prompt_segmentation.txt")
    parser.add_argument("--output-json", default="files/Documentos-Segmentados/resultado_anotacao.json")
    parser.add_argument("--model-name", default="gemini-2.0-flash")
    parser.add_argument("--text-column", default="texto_ner")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_stage3_segmentation(
        input_csv=args.input,
        prompt_file=args.prompt_file,
        output_json=args.output_json,
        model_name=args.model_name,
        text_column=args.text_column,
        id_column=args.id_column,
        max_docs=args.max_docs,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
    )
    print(summary)


if __name__ == "__main__":
    main()
