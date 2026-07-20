"""Tests de qa_llm_generate.py usando un cliente FAKE (sin pegarle a la API real ni
necesitar ANTHROPIC_API_KEY) -- solo se prueba la logica de extraccion/filtrado de
la respuesta de la tool, no la calidad real de lo que genera el modelo (eso se
verifica a mano, ver notebooks/diagnostico.ipynb seccion 9, corriendo qa_build.py contra la API real)."""

from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import pytest

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
    print(f"\n[test] pares extraidos: {len(pairs)}")
    for p in pairs:
        print(f"  - question={p.question!r} generation_detail={p.generation_detail!r}")
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
    print(f"\n[test] de 3 candidatos (1 valido, 2 incompletos) sobrevivieron: {len(pairs)}")
    assert len(pairs) == 1
    assert pairs[0].question == "¿Pregunta valida?"


def test_lista_vacia_si_el_modelo_no_encuentra_contenido_sustancioso():
    client = _fake_client_returning([])
    pairs = generate_llm_qa(project="p", section_path="s", chunk_text="TO DO.", max_pairs=3, model="m", client=client)
    print(f"\n[test] modelo devolvio lista vacia -> pairs = {pairs}")
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
    print(f"\n[test] lista con 1 string suelto + 1 par valido -> sobrevivieron: {len(pairs)}")
    assert len(pairs) == 1
    assert pairs[0].question == "¿Pregunta valida?"


# --- Reintentos ante errores transitorios --------------------------------------------


def _fake_status_error(status_code: int) -> anthropic.APIStatusError:
    """Construye una instancia REAL de anthropic.APIStatusError (no un mock) para
    probar _es_reintentable() contra el tipo exacto que lanza el SDK."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status_code=status_code, request=request)
    return anthropic.APIStatusError("fake", response=response, body=None)


def _fake_client_failing_then_succeeding(n_fallas: int, status_code: int, pairs: list[dict]):
    """Cliente falso que lanza un error (status_code dado) las primeras `n_fallas`
    veces, y a partir de ahi responde normalmente. Devuelve (cliente, contador) para
    poder verificar cuantas veces se intento de verdad."""
    intentos = {"n": 0}

    class _FakeMessages:
        def create(self, **kwargs):
            intentos["n"] += 1
            if intentos["n"] <= n_fallas:
                raise _fake_status_error(status_code)
            tool_use_block = SimpleNamespace(type="tool_use", input={"pairs": pairs})
            return SimpleNamespace(content=[tool_use_block])

    return SimpleNamespace(messages=_FakeMessages()), intentos


def test_reintenta_ante_error_transitorio_y_termina_ok(monkeypatch):
    """429 (rate limit) es reintentable -- si se resuelve antes de agotar los
    intentos, generate_llm_qa debe terminar devolviendo el resultado normal."""
    monkeypatch.setattr("docs2llm.qa_llm_generate.time.sleep", lambda *_: None)  # no esperar de verdad en el test
    client, intentos = _fake_client_failing_then_succeeding(
        n_fallas=2, status_code=429,
        pairs=[{"question": "¿Q?", "answer": "A", "evidence_span": "A", "answer_type": "extractive"}],
    )
    pairs = generate_llm_qa(project="p", section_path="s", chunk_text="...", max_pairs=3, model="m", client=client)
    print(f"\n[test] intentos realizados: {intentos['n']}")
    assert intentos["n"] == 3, "debio fallar 2 veces (429) y recien tener exito al 3er intento"
    assert len(pairs) == 1


def test_no_reintenta_ante_error_no_transitorio(monkeypatch):
    """400 (request mal formado) NO es reintentable -- reintentarlo no lo arregla,
    asi que debe propagar de inmediato, en el primer intento."""
    monkeypatch.setattr("docs2llm.qa_llm_generate.time.sleep", lambda *_: None)
    client, intentos = _fake_client_failing_then_succeeding(n_fallas=99, status_code=400, pairs=[])
    with pytest.raises(anthropic.APIStatusError):
        generate_llm_qa(project="p", section_path="s", chunk_text="...", max_pairs=3, model="m", client=client)
    print(f"\n[test] intentos realizados antes de propagar: {intentos['n']}")
    assert intentos["n"] == 1, "un error 400 (no reintentable) debe propagar en el primer intento, sin reintentar"


def test_agota_reintentos_y_propaga_si_el_error_persiste(monkeypatch):
    """Si un error reintentable (429) persiste mas alla de los intentos
    configurados, tiene que terminar propagando (no quedarse reintentando para
    siempre, ni fallar en silencio)."""
    monkeypatch.setattr("docs2llm.qa_llm_generate.time.sleep", lambda *_: None)
    client, intentos = _fake_client_failing_then_succeeding(n_fallas=99, status_code=429, pairs=[])
    with pytest.raises(anthropic.APIStatusError):
        generate_llm_qa(project="p", section_path="s", chunk_text="...", max_pairs=3, model="m", client=client)
    print(f"\n[test] intentos realizados: {intentos['n']}")
    assert intentos["n"] == 3, "debio agotar los 3 intentos configurados antes de propagar el error"
