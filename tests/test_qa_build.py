"""Tests de qa_build.py: aislamiento de errores por chunk al generar Q&A vía LLM
-- un chunk que falla (incluso agotando los reintentos de qa_llm_generate.py) no
debe tirar abajo el resto de la corrida de build-qa."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from docs2llm.config import QAConfig
from docs2llm.qa_build import _generate_candidates_for_row


@dataclasses.dataclass
class _FakeRow:
    """Doble minimo de CorpusRow -- solo los campos que _generate_candidates_for_row
    realmente usa."""

    id: str = "doc.md#0"
    project: str = "p"
    section_path: str = "s"
    text: str = "algo de prosa real, no placeholder"
    content_type: str = "prose"
    is_low_signal: bool = False
    duplicate_of: str | None = None


_QA_CONFIG = QAConfig(max_pairs_per_document=8, llm_model="claude-sonnet-5", enable_llm_judge=False)


class _ClienteQueSiempreFalla:
    class _Messages:
        def create(self, **kwargs):
            raise RuntimeError("fallo simulado de la API (no deberia importar cual)")

    messages = _Messages()


class _ClienteOk:
    class _Messages:
        def create(self, **kwargs):
            tool_use_block = SimpleNamespace(
                type="tool_use",
                input={"pairs": [{"question": "¿Q?", "answer": "A", "evidence_span": "A", "answer_type": "extractive"}]},
            )
            return SimpleNamespace(content=[tool_use_block])

    messages = _Messages()


def test_chunk_que_falla_devuelve_vacio_en_vez_de_propagar():
    """Un chunk cuya llamada al LLM revienta (cualquier excepcion, no solo las de
    anthropic) no debe propagar el error hacia build_qa_rows -- se trata como si no
    hubiera producido candidatos, y queda marcado con method='error' para que el
    reporte final (chunks_con_error_llm) lo pueda contar."""
    pairs, method = _generate_candidates_for_row(_FakeRow(), _QA_CONFIG, _ClienteQueSiempreFalla())
    print(f"\n[test] pairs={pairs}  method={method!r}")
    assert pairs == []
    assert method == "error"


def test_chunk_sin_error_sigue_devolviendo_llm_normal():
    """Contraste: sin fallas, el comportamiento normal (method='llm', candidatos
    reales) sigue exactamente igual que antes de agregar el aislamiento de errores."""
    pairs, method = _generate_candidates_for_row(_FakeRow(), _QA_CONFIG, _ClienteOk())
    print(f"\n[test] pairs={pairs}  method={method!r}")
    assert method == "llm"
    assert len(pairs) == 1
    assert pairs[0].question == "¿Q?"
