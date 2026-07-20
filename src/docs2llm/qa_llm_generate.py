"""Genera preguntas y respuestas usando la API de Claude, para chunks de
PROSA/reglas de negocio -- el contenido que qa_templates.py no puede cubrir
porque requiere "entender" texto libre, en vez de sacar un campo que ya viene
estructurado (ver la explicacion al principio de qa_templates.py para la mitad
determinista del enfoque hibrido).

Necesita tener configurada la variable ANTHROPIC_API_KEY (ver .env.example y el
README) -- se lee a traves de `anthropic.Anthropic()` (el cliente la busca solo
en las variables de entorno), nunca se escribe directo en el codigo ni se pasa
como texto plano por ningun lado.

Se usa "tool use" (una forma de la API donde el modelo devuelve datos en un
formato fijo, no texto libre) en vez de pedirle al modelo que devuelva JSON como
texto suelto: esto fuerza una salida con una estructura fija, sin tener que
lidiar con bloques de codigo Markdown alrededor del JSON, o texto explicativo
antes/despues que rompa un `json.loads()` ingenuo.

Reintentos ante errores transitorios: build-qa hace una llamada separada por
cada chunk de prosa (unos ~250 en este corpus). El cliente de `anthropic` ya
reintenta un par de veces por su cuenta ante fallas transitorias (su
`max_retries` por defecto es 2), pero para una corrida larga y secuencial eso
puede no alcanzar (un rate limit sostenido, por ejemplo, se resuelve solo
esperando mas de lo que esos 2 reintentos por defecto cubren). Por eso
`generate_llm_qa` agrega una capa extra de reintento con espera creciente,
pero SOLO para errores que tiene sentido reintentar (ver `_es_reintentable`,
mas abajo) -- reintentar un request mal formado o una clave invalida no lo
arregla, solo demora el error. El aislamiento de que UN chunk que falla no
tire abajo toda la corrida de build-qa vive un nivel mas arriba, en
qa_build.py (ver ahi el por que).
"""

from __future__ import annotations

import time

# El SDK oficial de Anthropic: la libreria que arma y envia las llamadas HTTP a
# la API de Claude, y convierte la respuesta en objetos de Python faciles de usar.
import anthropic

from docs2llm.qa_units import QAPair

# Cuantas veces se intenta una llamada ante un error transitorio, y cuanto se
# espera entre intentos (crece exponencial: 2s, 4s, ... para no bombardear la
# API si el problema es un rate limit real). 3 intentos en total es un punto
# medio: suficiente para tolerar una falla puntual sin alargar demasiado una
# corrida de ~250 llamadas si el problema resulta ser persistente.
_MAX_INTENTOS = 3
_ESPERA_BASE_SEGUNDOS = 2.0

# Codigos HTTP donde reintentar tiene sentido: limite de tasa (429) y errores
# del lado del servidor (5xx, incluido 529 "overloaded" especifico de Anthropic).
# CUALQUIER OTRO status (400 request mal formado, 401/403 auth, 404, 422) es un
# problema que reintentar no arregla -- se deja pasar de inmediato.
_STATUS_CODES_REINTENTABLES = {429, 500, 502, 503, 529}


def _es_reintentable(exc: Exception) -> bool:
    """True si vale la pena reintentar `exc`: un error de conexion (red caida,
    timeout) o un status HTTP transitorio (ver _STATUS_CODES_REINTENTABLES)."""
    if isinstance(exc, anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in _STATUS_CODES_REINTENTABLES
    return False

_SYSTEM_PROMPT = """Eres un asistente que genera datasets de pregunta-respuesta de alta calidad \
para entrenar y evaluar modelos de lenguaje sobre documentacion tecnica interna de una empresa.

Reglas estrictas:
- Cada pregunta debe poder responderse usando EXCLUSIVAMENTE la informacion del fragmento \
que se te da. No inventes datos que no esten ahi, ni completes con conocimiento general.
- Cada respuesta debe venir acompañada de "evidence_span": una cita LITERAL (copiada \
palabra por palabra, sin parafrasear ni resumir) del fragmento, que sustente completamente \
la respuesta. Si no puedes citar una porcion literal que la sustente, no generes ese par.
- Prioriza datos concretos y verificables: reglas de negocio, limites numericos, nombres \
de campos, comportamientos especificos, codigos de error, ejemplos de valores. Evita \
preguntas genericas tipo "de que trata este documento" o "que es X" cuando X no se define \
con precision en el fragmento.
- Si el fragmento es puro boilerplate, placeholder, o no tiene contenido sustancioso para \
una pregunta seria, devuelve una lista de pares VACIA en vez de forzar una de baja calidad."""

_QA_TOOL = {
    "name": "submit_qa_pairs",
    "description": "Envia los pares de pregunta-respuesta generados a partir del fragmento de documentacion dado.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "evidence_span": {
                            "type": "string",
                            "description": "Cita textual EXACTA (subcadena literal, sin parafrasear) del fragmento fuente que sustenta la respuesta.",
                        },
                        "answer_type": {
                            "type": "string",
                            "enum": ["extractive", "generative", "numeric", "boolean"],
                        },
                    },
                    "required": ["question", "answer", "evidence_span", "answer_type"],
                },
            }
        },
        "required": ["pairs"],
    },
}


def _build_user_message(project: str, section_path: str, chunk_text: str, max_pairs: int) -> str:
    return (
        f"Proyecto: {project}\n"
        f"Sección del documento: {section_path}\n\n"
        f'Fragmento:\n"""\n{chunk_text}\n"""\n\n'
        f"Genera hasta {max_pairs} pares de pregunta-respuesta siguiendo las reglas del system prompt."
    )


def generate_llm_qa(
    *,
    project: str,
    section_path: str,
    chunk_text: str,
    max_pairs: int,
    model: str,
    client: anthropic.Anthropic | None = None,
) -> list[QAPair]:
    """Genera candidatos de pregunta-respuesta para un chunk, usando la API de Claude.

    El parametro `client` se puede pasar desde afuera a proposito: qa_build.py
    crea UNA sola instancia y la reutiliza para todos los chunks (asi evita abrir
    una conexion HTTP nueva por cada llamada), y los tests pueden pasar un cliente
    de mentira sin necesitar pegarle a la API real (ver tests/test_llm_generate.py,
    que usa un doble de prueba en vez de una clave real).
    """
    client = client or anthropic.Anthropic()

    response = None
    for intento in range(1, _MAX_INTENTOS + 1):
        try:
            response = client.messages.create(
                model=model,
                # 1536 es un presupuesto de salida, no un limite de entrada: tiene que
                # alcanzar para el JSON estructurado de hasta `max_pairs` pares (question +
                # answer + evidence_span + answer_type cada uno, tipicamente 100-200 tokens
                # por par en este corpus) sin cortar la respuesta a la mitad. Se eligio con
                # margen sobre esa estimacion en vez de calcularlo exacto por llamada: un
                # limite generoso y fijo es mas simple, y el costo de sobrar tokens de
                # salida no usados es minimo comparado con el riesgo de una respuesta
                # truncada (que rompería el parseo del tool_use).
                max_tokens=1536,
                system=_SYSTEM_PROMPT,
                tools=[_QA_TOOL],
                tool_choice={"type": "tool", "name": "submit_qa_pairs"},
                messages=[{"role": "user", "content": _build_user_message(project, section_path, chunk_text, max_pairs)}],
            )
            break
        except Exception as exc:
            if not _es_reintentable(exc) or intento == _MAX_INTENTOS:
                raise
            espera = _ESPERA_BASE_SEGUNDOS * (2 ** (intento - 1))
            print(
                f"[qa_llm_generate] {project} / {section_path}: intento {intento}/{_MAX_INTENTOS} "
                f"fallo ({exc.__class__.__name__}), reintentando en {espera:.0f}s..."
            )
            time.sleep(espera)

    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return []  # el modelo no llamo a la herramienta (raro cuando se fuerza con tool_choice, pero se contempla igual)

    raw_pairs = tool_use_block.input.get("pairs", [])
    return [
        QAPair(
            question=str(p["question"]).strip(),
            answer=str(p["answer"]).strip(),
            evidence_span=str(p["evidence_span"]),
            answer_type=str(p.get("answer_type", "generative")),
            generation_detail=f"llm:{model}",
        )
        for p in raw_pairs
        # El chequeo isinstance() protege contra items mal formados: a pesar de
        # que el input_schema exige que "pairs" sea una lista de objetos, se vio
        # en una corrida real contra la API que el modelo a veces mete un texto
        # suelto en la lista en vez de un objeto -- sin este chequeo, p.get()
        # revienta con un error (AttributeError) porque un texto no tiene ese metodo.
        if isinstance(p, dict) and p.get("question") and p.get("answer") and p.get("evidence_span")
    ]
