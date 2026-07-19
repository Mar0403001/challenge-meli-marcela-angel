"""Tests de qa_llm_generate.py usando un cliente FAKE (sin pegarle a la API real ni
necesitar ANTHROPIC_API_KEY) -- solo se prueba la logica de extraccion/filtrado de
la respuesta de la tool, no la calidad real de lo que genera el modelo (eso se
verifica a mano, ver notebooks/diagnostico.ipynb seccion 9, corriendo qa_build.py contra la API real)."""

from __future__ import annotations

from types import SimpleNamespace

from docs2llm.qa_llm_generate import generate_llm_qa


def _fake_client_returning(pairs: list[dict]):
    """Stub minimo que imita la forma de client.messages.create(...) de anthropic:
    un objeto con .content = [bloque_tool_use]."""

    class _FakeMessages:
        def create(self, **kwargs):
            tool_use_block = SimpleNamespace(type="tool_use", input={"pairs": pairs})
            return SimpleNamespace(content=[tool_use_block])

    return SimpleNamespace(messages=_FakeMessages())


def test_extrae_pares_validos_del_tool_use():
    client = _fake_client_returning(
        [
            {
                "question": "¿Cuál es el límite de rate-limit para países RetailHub?",
                "answer": "1000 solicitudes cada 5 minutos",
                "evidence_span": "1000 solicitudes cada 5 minutos",
                "answer_type": "numeric",
            }
        ]
    )
    pairs = generate_llm_qa(
        project="traffic-gate-api", section_path="Reglas > Scope Front", chunk_text="...",
        max_pairs=3, model="claude-sonnet-5", client=client,
    )
    assert len(pairs) == 1
    assert pairs[0].question.startswith("¿Cuál es el límite")
    assert pairs[0].generation_detail == "llm:claude-sonnet-5"


def test_descarta_pares_incompletos():
    """Un par sin evidence_span (o sin question/answer) no debe pasar -- aunque el
    modelo lo devuelva, no es un par usable para grounding."""
    client = _fake_client_returning(
        [
            {"question": "¿Pregunta valida?", "answer": "Respuesta", "evidence_span": "Respuesta", "answer_type": "extractive"},
            {"question": "", "answer": "sin pregunta", "evidence_span": "algo", "answer_type": "extractive"},
            {"question": "sin evidencia", "answer": "algo", "evidence_span": "", "answer_type": "extractive"},
        ]
    )
    pairs = generate_llm_qa(project="p", section_path="s", chunk_text="...", max_pairs=3, model="m", client=client)
    assert len(pairs) == 1
    assert pairs[0].question == "¿Pregunta valida?"


def test_lista_vacia_si_el_modelo_no_encuentra_contenido_sustancioso():
    client = _fake_client_returning([])
    pairs = generate_llm_qa(project="p", section_path="s", chunk_text="TO DO.", max_pairs=3, model="m", client=client)
    assert pairs == []


def test_item_malformado_no_dict_se_ignora_sin_reventar():
    """Pese al input_schema (pairs -> array de object), se vio en una corrida real
    contra la API que el modelo a veces mete un string suelto en la lista de pairs
    en vez de un objeto -- no debe tirar AttributeError, solo ignorar ese item."""
    client = _fake_client_returning(
        [
            "un string suelto en vez de un objeto",
            {"question": "¿Pregunta valida?", "answer": "Respuesta", "evidence_span": "Respuesta", "answer_type": "extractive"},
        ]
    )
    pairs = generate_llm_qa(project="p", section_path="s", chunk_text="...", max_pairs=3, model="m", client=client)
    assert len(pairs) == 1
    assert pairs[0].question == "¿Pregunta valida?"
