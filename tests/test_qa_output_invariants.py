"""Valida los invariantes de calidad sobre los artefactos YA GENERADOS y committeados
en data/processed/ (corpus.jsonl y qa.jsonl) -- no vuelve a llamar a la API de Claude
ni a docs_raw/, para que este test corra siempre (sin red, sin API key) y sirva como
prueba de que el dataset ENTREGADO cumple lo que promete, no solo el codigo en abstracto.

Si estos archivos no existen todavia (clon nuevo, antes de correr build-corpus/build-qa),
el test se salta en vez de fallar -- no es un requisito para que el codigo funcione,
es una verificacion extra sobre el resultado ya producido.
"""

from __future__ import annotations

import json
import re

import pytest

from docs2llm.config import REPO_ROOT

_CORPUS_PATH = REPO_ROOT / "data" / "processed" / "corpus.jsonl"
_QA_PATH = REPO_ROOT / "data" / "processed" / "qa.jsonl"

pytestmark = pytest.mark.skipif(
    not (_CORPUS_PATH.exists() and _QA_PATH.exists()),
    reason="corpus.jsonl/qa.jsonl no generados todavia -- correr build-corpus y build-qa primero",
)


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="module")
def corpus_rows() -> dict[str, dict]:
    rows = [json.loads(line) for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n[fixture] corpus_rows: {len(rows)} filas leidas de {_CORPUS_PATH.name}")
    return {row["id"]: row for row in rows}


@pytest.fixture(scope="module")
def qa_rows() -> list[dict]:
    rows = [json.loads(line) for line in _QA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\n[fixture] qa_rows: {len(rows)} filas leidas de {_QA_PATH.name}")
    return rows


def test_todas_las_filas_de_qa_pasaron_qc(qa_rows):
    n_ok = sum(1 for row in qa_rows if row["qc_passed"])
    print(f"[test] {n_ok}/{len(qa_rows)} filas con qc_passed=True")
    assert all(row["qc_passed"] for row in qa_rows), "qa.jsonl no deberia contener filas que fallaron control de calidad"


def test_evidence_span_es_subcadena_literal_del_chunk_de_origen(qa_rows, corpus_rows):
    for i, row in enumerate(qa_rows):
        chunk = corpus_rows.get(row["source_chunk_id"])
        assert chunk is not None, f"source_chunk_id inexistente en corpus.jsonl: {row['source_chunk_id']}"
        assert _collapse(row["evidence_span"]) in _collapse(chunk["text"]), (
            f"evidence_span de {row['id']} no es una subcadena literal de su chunk fuente -- posible alucinacion sin detectar"
        )
    print(f"[test] evidence_span verificado como subcadena literal en las {len(qa_rows)} filas")


def test_split_de_qa_coincide_siempre_con_el_split_del_chunk_de_origen(qa_rows, corpus_rows):
    for row in qa_rows:
        chunk = corpus_rows[row["source_chunk_id"]]
        assert row["split"] == chunk["split"], (
            f"{row['id']}: split de qa.jsonl ({row['split']}) no coincide con el de su chunk de origen ({chunk['split']})"
        )
    print(f"[test] split heredado correctamente en las {len(qa_rows)} filas")


def test_no_hay_filas_de_qa_derivadas_de_chunks_marcados_como_duplicado(qa_rows, corpus_rows):
    for row in qa_rows:
        chunk = corpus_rows[row["source_chunk_id"]]
        assert chunk["duplicate_of"] is None, (
            f"{row['id']} se genero desde un chunk marcado como duplicado ({chunk['id']}) -- no deberia generarse Q&A ahi"
        )
    print(f"[test] ninguna de las {len(qa_rows)} filas viene de un chunk duplicado")
