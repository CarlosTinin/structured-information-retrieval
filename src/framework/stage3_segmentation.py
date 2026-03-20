from __future__ import annotations

import argparse
import ast
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

    def _normalize_payload(payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            return {"total_sentencas": len(payload), "dados": payload}
        if isinstance(payload, dict):
            if "dados" in payload and isinstance(payload["dados"], list):
                if "total_sentencas" not in payload:
                    payload["total_sentencas"] = len(payload["dados"])
                return payload
            return {"total_sentencas": 0, "dados": []}
        raise ValueError("JSON com tipo inválido para payload")

    def _repair_common_json_issues(candidate: str) -> str:
        repaired = candidate
        # Remove vírgulas sobrando antes de ] ou }
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
        # Insere vírgula faltante entre objetos consecutivos: } { -> }, {
        repaired = re.sub(r"}\s*{", "},{", repaired)
        return repaired

    candidates = [cleaned]
    candidates.append(_repair_common_json_issues(cleaned))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return _normalize_payload(payload)
        except json.JSONDecodeError:
            pass

    # fallback extra: tenta interpretar como literal Python seguro
    for candidate in candidates:
        try:
            payload = ast.literal_eval(candidate)
            return _normalize_payload(payload)
        except Exception:
            pass

    # fallback: tenta encontrar primeiro bloco JSON no texto
    block_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if block_match:
        candidate = block_match.group(1)
        for variant in (candidate, re.sub(r",\s*([\]}])", r"\1", candidate), re.sub(r"}\s*{", "},{", candidate)):
            try:
                payload = json.loads(variant)
                return _normalize_payload(payload)
            except json.JSONDecodeError:
                continue
            except Exception:
                continue

        for variant in (candidate, re.sub(r",\s*([\]}])", r"\1", candidate), re.sub(r"}\s*{", "},{", candidate)):
            try:
                payload = ast.literal_eval(variant)
                return _normalize_payload(payload)
            except Exception:
                continue

    raise ValueError("Resposta da LLM não contém JSON válido no formato esperado.")


def _response_to_text(response: Any) -> str:
    direct_text = getattr(response, "text", "")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text

    candidates = getattr(response, "candidates", None)
    if candidates:
        parts_text: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for part in parts:
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt.strip():
                    parts_text.append(txt)
        if parts_text:
            return "\n".join(parts_text)

    return ""


def _build_prompt(base_prompt: str, text: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "==================== TEXTO PARA ANOTAÇÃO ====================\n"
        f"{text}"
    )


def _per_doc_output_path(output_json: str, doc_index: int) -> Path:
    output_path = Path(output_json)
    out_dir = output_path.parent
    return out_dir / f"resultado_anotacao_{doc_index:02d}.json"


def run_stage3_segmentation(
    input_csv: str,
    prompt_file: str,
    output_json: str,
    model_name: str = "gemini-2.5-flash-lite",
    text_column: str = "texto_ner",
    id_column: str = "id",
    filter_label_column: str = "decisao",
    filter_label_value: str = "condenação",
    max_docs: int | None = None,
    sleep_seconds: float = 0.0,
    api_key_env: str = "GEMINI_API_KEY",
    request_timeout: int = 180,
    max_retries: int = 3,
    retry_backoff_seconds: float = 3.0,
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

    total_lidos = len(df)
    if filter_label_column and filter_label_column in df.columns and filter_label_value:
        target = filter_label_value.strip().lower()
        df = df[df[filter_label_column].astype(str).str.strip().str.lower() == target].copy()
        print(
            f"Filtro aplicado em {filter_label_column}='{filter_label_value}': "
            f"{total_lidos} -> {len(df)} documentos"
        )
    else:
        print("Filtro de tipo não aplicado (coluna ausente ou valor vazio).")

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
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"},
                    request_options={"timeout": request_timeout},
                )
                response_text = _response_to_text(response)
                payload = _extract_json_payload(response_text)
                result_item = {
                    "index": idx,
                    "doc_id": doc_id,
                    "total_sentencas": int(payload.get("total_sentencas", len(payload.get("dados", [])))),
                    "dados": payload.get("dados", []),
                }
                results.append(result_item)

                per_doc_payload = {
                    "index": idx,
                    "doc_id": doc_id,
                    "model_name": model_name,
                    "total_sentencas": result_item["total_sentencas"],
                    "dados": result_item["dados"],
                }
                per_doc_path = _per_doc_output_path(output_json, len(results))
                ensure_parent_dir(str(per_doc_path))
                save_json(per_doc_payload, str(per_doc_path))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                err_text = str(exc).lower()
                transient = any(token in err_text for token in ["504", "timeout", "deadline", "temporar", "429", "unavailable"])
                if attempt < max_retries and transient:
                    wait_s = retry_backoff_seconds * attempt
                    print(f"[Aviso] Falha transitória no doc {doc_id} (tentativa {attempt}/{max_retries}): {exc}")
                    print(f"[Aviso] Aguardando {wait_s:.1f}s para retry...")
                    time.sleep(wait_s)
                    continue
                break

        if last_error is not None:
            err_msg = str(last_error)
            if "json válido" in err_msg.lower() and 'response' in locals():
                try:
                    debug_excerpt = _response_to_text(response)
                    if not debug_excerpt:
                        debug_excerpt = str(response)
                    debug_excerpt = debug_excerpt.replace("\n", " ")[:400]
                    err_msg = f"{err_msg} | resposta_bruta={debug_excerpt}"
                except Exception:
                    pass
            errors.append(
                {
                    "index": idx,
                    "doc_id": doc_id,
                    "error": err_msg,
                }
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    output_payload = {
        "model_name": model_name,
        "input_csv": input_csv,
        "text_column": text_column,
        "filter_label_column": filter_label_column,
        "filter_label_value": filter_label_value,
        "total_lidos": total_lidos,
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
        "output_dir": str(Path(output_json).parent),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 3 - Segmentação com LLM (Gemini)")
    parser.add_argument("--input", default="files/output/dataset_normalized_for_ner.csv")
    parser.add_argument("--prompt-file", default="src/prompts/prompt_segmentation.txt")
    parser.add_argument("--output-json", default="files/Documentos-Segmentados/resultado_anotacao.json")
    parser.add_argument("--model-name", default="gemini-2.5-flash-lite")
    parser.add_argument("--text-column", default="texto_ner")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--filter-label-column", default="decisao")
    parser.add_argument("--filter-label-value", default="condenação")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0)
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
        filter_label_column=args.filter_label_column,
        filter_label_value=args.filter_label_value,
        max_docs=args.max_docs,
        sleep_seconds=args.sleep_seconds,
        api_key_env=args.api_key_env,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    print(summary)


if __name__ == "__main__":
    main()
