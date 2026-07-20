"""Conecta la generacion de preguntas y respuestas: lee corpus.jsonl (NO vuelve a
correr todo el pipeline del corpus), genera candidatos (plantillas para tablas y
OpenAPI, un modelo de lenguaje para prosa), les aplica el control de calidad, les
aplica el tope anti-volumen por documento, y devuelve las filas finales de qa.jsonl.

Por que lee corpus.jsonl en vez de recibir los CorpusRow ya en memoria: esto es lo
que garantiza, de forma estructural, que el `split` de cada fila de preguntas sea EXACTAMENTE el mismo que ya
se decidio para su chunk de origen -- nunca se calcula un split propio y distinto
para las preguntas (ver notebooks/diagnostico.ipynb, seccion 7).

Por que un chunk que falla no tira abajo toda la corrida: build-qa hace una
llamada a la API por cada chunk de prosa (unos ~250 en este corpus), una por
una, y las filas finales solo se escriben a disco al terminar TODO el lote (ver
cli.py). Sin aislar el error por chunk, una sola falla que sobrevive a los
reintentos de qa_llm_generate.py (rate limit sostenido, error no reintentable,
lo que sea) perderia el trabajo ya hecho sobre los demas ~250 chunks -- carisimo
en tiempo y en llamadas ya pagadas a la API. Por eso `_generate_candidates_for_row`
atrapa cualquier excepcion de ese chunk puntual, la deja loggeada con claridad
(ver `chunks_con_error_llm` en el reporte de build_qa_rows), y sigue con el resto
en vez de propagar el error hacia arriba.
"""

from __future__ import annotations

import dataclasses
import sys
from collections import Counter, defaultdict

import anthropic

from docs2llm.config import QAConfig
from docs2llm.corpus_build import CorpusRow
from docs2llm.qa_grounding import QualityChecks, check_evidence, check_length, find_near_duplicate_question_indices
from docs2llm.qa_llm_generate import generate_llm_qa
from docs2llm.qa_templates import generate_template_qa
from docs2llm.qa_units import QAPair


@dataclasses.dataclass
class QARow:
    id: str
    project: str
    project_version: str
    source_path: str
    section_path: str
    source_chunk_id: str
    question: str
    answer: str
    answer_type: str
    evidence_span: str
    generation_method: str  # "template" | "llm"
    generation_detail: str
    quality_checks: dict
    qc_passed: bool
    split: str  # heredado directo de corpus_row.split, nunca se vuelve a calcular (ver arriba)


# Orden de prioridad al recortar por el tope anti-volumen (max_pairs_per_document):
# se prefieren las respuestas mas "verificables" (un valor cerrado tipo enum, o un
# numero concreto, son mas dificiles de inventar por error y mas faciles de
# evaluar automaticamente que una respuesta abierta generada por un modelo de
# lenguaje). 
_ANSWER_TYPE_PRIORITY = {"enum": 0, "numeric": 1, "boolean": 1, "extractive": 2, "generative": 3}


def _generate_candidates_for_row(row: CorpusRow, qa_config: QAConfig, llm_client) -> tuple[list[QAPair], str]:
    """Devuelve (candidatos, metodo_de_generacion) para UN chunk. Decide de donde
    sacar las preguntas segun el content_type -- ver notebooks/diagnostico.ipynb,
    seccion 9."""
    if row.duplicate_of is not None:
        # No tiene sentido generar (ni potencialmente pagar una llamada a la API
        # por) un chunk que ya se sabe que es un duplicado exacto de otro -- el
        # chunk original ya cubre esa misma pregunta.
        return [], "n/a"

    if row.content_type in ("table", "api_spec"):
        return generate_template_qa(row.text, row.content_type, row.section_path), "template"

    if row.content_type == "prose" and not row.is_low_signal and llm_client is not None:
        try:
            pairs = generate_llm_qa(
                project=row.project,
                section_path=row.section_path,
                chunk_text=row.text,
                max_pairs=qa_config.max_pairs_per_document,
                model=qa_config.llm_model,
                client=llm_client,
            )
        except Exception as exc:
            # Este chunk puntual no pudo generar Q&A (agoto los reintentos de
            # qa_llm_generate.py, o fue un error no reintentable) -- se trata
            # como si no hubiera producido candidatos, en vez de tirar abajo el
            # resto de la corrida. Se deja un rastro explicito en stderr y en el reporte
            # final (chunks_con_error_llm, ver build_qa_rows) para que la falla
            # nunca quede en silencio.
            print(f"[qa_build] AVISO: {row.id} no genero Q&A ({exc.__class__.__name__}: {exc}) -- se continua con el resto.", file=sys.stderr)
            return [], "error"
        return pairs, "llm"

    return [], "n/a"  # "code", o prosa marcada is_low_signal: sin fuente de preguntas en este MVP


def _select_top_pairs(items: list[tuple], max_per_doc: int) -> list[tuple]:
    """Reparte en ronda (round-robin) entre los distintos chunks de un mismo
    documento, para que el tope anti-volumen no se llene entero con los pares del
    primer chunk nada mas -- dentro de cada chunk, se prioriza segun
    _ANSWER_TYPE_PRIORITY (ver arriba)."""
    by_chunk: dict[int, list] = defaultdict(list)
    for item in items:
        row = item[0]
        by_chunk[row.chunk_index].append(item)
    for bucket in by_chunk.values():
        bucket.sort(key=lambda item: _ANSWER_TYPE_PRIORITY.get(item[1].answer_type, 4))

    chunk_indices = sorted(by_chunk)
    selected: list = []
    round_robin_pos = 0
    while len(selected) < max_per_doc and any(by_chunk[ci] for ci in chunk_indices):
        ci = chunk_indices[round_robin_pos % len(chunk_indices)]
        if by_chunk[ci]:
            selected.append(by_chunk[ci].pop(0))
        round_robin_pos += 1
    return selected


def build_qa_rows(
    corpus_rows: list[CorpusRow], qa_config: QAConfig, llm_client: anthropic.Anthropic | None = None
) -> tuple[list[QARow], dict]:
    """Genera los candidatos (plantillas + modelo de lenguaje), les aplica el
    control de calidad y el tope anti-volumen, y devuelve las filas finales de
    qa.jsonl junto con un reporte del proceso.

    Por que el control de calidad corre en 2 pasadas (ver el cuerpo de la
    funcion): el chequeo de evidencia y el de longitud son independientes, se
    pueden evaluar candidato por candidato; pero detectar preguntas
    casi-duplicadas necesita ver de una vez todo el lote de un mismo proyecto
    (ver qa_grounding.find_near_duplicate_question_indices).
    """
    # 1. Generar TODOS los candidatos (las plantillas no cuestan nada; el modelo
    # de lenguaje solo se llama para prosa que no es de bajo contenido, ver
    # _generate_candidates_for_row). Un chunk que falla (method == "error", ver
    # _generate_candidates_for_row) no aporta candidatos pero tampoco corta el
    # loop -- se cuenta aparte para que quede visible en el reporte final.
    candidates: list[tuple[CorpusRow, QAPair, str]] = []
    n_chunks_con_error_llm = 0
    for row in corpus_rows:
        pairs, method = _generate_candidates_for_row(row, qa_config, llm_client)
        if method == "error":
            n_chunks_con_error_llm += 1
        for pair in pairs:
            candidates.append((row, pair, method))

    # 2. Chequeo de evidencia + longitud (no dependen del resto del lote, se
    # pueden evaluar cada uno por su cuenta).
    checked_evidence_length: list[tuple[CorpusRow, QAPair, str, bool, bool]] = [
        (row, pair, method, check_evidence(pair.evidence_span, row.text), check_length(pair.question, pair.answer))
        for row, pair, method in candidates
    ]

    # 3. Preguntas casi-duplicadas DENTRO del mismo proyecto (necesita ver el lote completo).
    questions_by_project = [(row.project, pair.question) for row, pair, *_ in checked_evidence_length]
    duplicate_indices = find_near_duplicate_question_indices(questions_by_project)

    with_qc: list[tuple[CorpusRow, QAPair, str, QualityChecks]] = []
    for i, (row, pair, method, ev_ok, len_ok) in enumerate(checked_evidence_length):
        qc = QualityChecks(evidence_verified=ev_ok, length_ok=len_ok, not_near_duplicate_question=(i not in duplicate_indices))
        with_qc.append((row, pair, method, qc))

    passed = [item for item in with_qc if item[3].qc_passed]

    # 4. Anti-volumen: un tope de pares por DOCUMENTO (no por chunk), para que un
    # archivo con muchos chunks estructurados (por ejemplo, un swagger.yaml con
    # 35 secciones) no termine dominando el dataset final solo por volumen -- ver
    # "mas volumen no es mejor" en el enunciado del desafio.
    by_doc: dict[str, list] = defaultdict(list)
    for item in passed:
        by_doc[item[0].doc_id].append(item)

    final_rows: list[QARow] = []
    per_row_counter: Counter[str] = Counter()
    for doc_id in sorted(by_doc):  # se ordena para que el resultado sea siempre igual, sin depender del orden de un diccionario
        selected = _select_top_pairs(by_doc[doc_id], qa_config.max_pairs_per_document)
        for row, pair, method, qc in selected:
            k = per_row_counter[row.id]
            per_row_counter[row.id] += 1
            final_rows.append(
                QARow(
                    id=f"{row.id}::qa{k}",
                    project=row.project,
                    project_version=row.project_version,
                    source_path=row.source_path,
                    section_path=row.section_path,
                    source_chunk_id=row.id,
                    question=pair.question,
                    answer=pair.answer,
                    answer_type=pair.answer_type,
                    evidence_span=pair.evidence_span,
                    generation_method=method,
                    generation_detail=pair.generation_detail,
                    quality_checks=dataclasses.asdict(qc),
                    qc_passed=qc.qc_passed,
                    split=row.split,  # heredado, nunca se vuelve a calcular
                )
            )

    report = {
        "candidatos_generados": len(candidates),
        "candidatos_por_metodo": dict(Counter(method for _, _, method in candidates)),
        "candidatos_que_pasaron_qc": len(passed),
        "filas_finales_tras_tope_anti_volumen": len(final_rows),
        "filas_por_split": dict(Counter(r.split for r in final_rows)),
        "filas_por_metodo": dict(Counter(r.generation_method for r in final_rows)),
        # Chunks de prosa donde la llamada al LLM fallo (agoto los reintentos de
        # qa_llm_generate.py, o fue un error no reintentable) y se salteo en vez
        # de tirar abajo toda la corrida -- ver el docstring de este modulo.
        "chunks_con_error_llm": n_chunks_con_error_llm,
    }
    return final_rows, report
