"""Genera preguntas y respuestas de forma DETERMINISTA (sin usar un LLM) a partir
de contenido ya estructurado: tablas de Markdown (content_type == "table") y
chunks que vienen de una especificacion OpenAPI (content_type == "api_spec").

Por que esto va separado de qa_llm_generate.py: cuando la fuente ya viene
estructurada, la respuesta se puede sacar directo (un campo de una tabla, un
valor permitido de un parametro de OpenAPI) sin necesitar pedirle a un modelo de
lenguaje que "entienda" el texto -- la respuesta queda perfectamente respaldada
por como esta construido el proceso (la respuesta ES, literalmente, parte de la
fuente), y no cuesta nada de computo. Esta es la mitad "barata" del enfoque
hibrido: el modelo de lenguaje se reserva para prosa y reglas de negocio, que es
donde de verdad hace falta (ver qa_llm_generate.py).

Todas las funciones de este archivo trabajan sobre `chunk.text` (el texto que ya
quedo armado en corpus.jsonl), no vuelven a leer swagger.yaml ni el .md original:
las funciones de parse_openapi.py escriben un formato de texto conocido y fijo
(que nosotros mismos definimos), asi que se puede volver a sacar la informacion
con expresiones regulares apuntadas a ESE formato especifico, en vez de tener que
abrir y volver a leer el archivo original. Esto hace que la generacion de
preguntas no dependa para nada de docs_raw/: solo necesita corpus.jsonl.
"""

from __future__ import annotations

import re

from docs2llm.qa_units import QAPair


# --- Plantillas para chunks de OpenAPI (content_type == "api_spec") -----------------
# Estos patrones calzan EXACTAMENTE con el formato que escribe parse_openapi.py
# (en _format_operation / _format_schema) -- si ese formato cambia, estas
# expresiones regulares se tienen que actualizar junto con el (los dos lados de
# este acoplamiento los controlamos nosotros mismos, a proposito).

_ENDPOINT_LINE_RE = re.compile(r"^Endpoint: (\S+) (.+)$", re.MULTILINE)
_SUMMARY_LINE_RE = re.compile(r"^Summary: (.+)$", re.MULTILINE)
_PARAM_LINE_RE = re.compile(
    r"^  - (\S+) \(in: (\w+), type: (\w+)\)(?: enum=(\[.*?\]))?( \[required\])?: (.*)$", re.MULTILINE
)
_RESPONSE_LINE_RE = re.compile(r"^  - (\d{3}): (.*)$", re.MULTILINE)
_SCHEMA_HEADER_RE = re.compile(r"^Schema: (.+)$", re.MULTILINE)
_SCHEMA_TYPE_RE = re.compile(r"^Type: (.+)$", re.MULTILINE)
_SCHEMA_PROP_LINE_RE = re.compile(r"^  - (\S+) \((.+?)\)(?: enum=(\[.*?\]))?: (.*)$", re.MULTILINE)


def _qa_from_openapi_operation(text: str) -> list[QAPair]:
    """Genera preguntas sobre un chunk que empieza con "Endpoint: ..." (el
    resumen, el tipo/valores permitidos de cada parametro, el significado de cada
    respuesta HTTP), usando expresiones regulares sobre el formato fijo que
    escribe parse_openapi.py -- el respaldo de la respuesta (grounding) queda
    garantizado por como esta construido esto.
    """
    pairs: list[QAPair] = []

    endpoint_match = _ENDPOINT_LINE_RE.search(text)
    if not endpoint_match:
        return pairs
    method, path = endpoint_match.groups()
    endpoint_label = f"{method} {path}"

    summary_match = _SUMMARY_LINE_RE.search(text)
    if summary_match:
        line = summary_match.group(0)
        pairs.append(
            QAPair(
                question=f"¿Qué hace el endpoint `{endpoint_label}`?",
                answer=summary_match.group(1).strip(),
                evidence_span=line,
                answer_type="extractive",
                generation_detail="openapi_operation_summary",
            )
        )

    for name, location, type_, enum, required, desc in _PARAM_LINE_RE.findall(text):
        line_match = next(m for m in _PARAM_LINE_RE.finditer(text) if m.group(1) == name)
        line = line_match.group(0)
        req_txt = " (requerido)" if required else " (opcional)"
        pairs.append(
            QAPair(
                question=f"¿Qué tipo de dato tiene el parámetro `{name}` ({location}) del endpoint `{endpoint_label}`?",
                answer=f"{type_}{req_txt}. {desc}".strip(),
                evidence_span=line,
                answer_type="extractive",
                generation_detail="openapi_operation_parameter_type",
            )
        )
        if enum:
            pairs.append(
                QAPair(
                    question=f"¿Cuáles son los valores válidos para el parámetro `{name}` del endpoint `{endpoint_label}`?",
                    answer=enum,
                    evidence_span=line,
                    answer_type="enum",
                    generation_detail="openapi_operation_parameter_enum",
                )
            )

    for status, desc in _RESPONSE_LINE_RE.findall(text):
        line_match = next(m for m in _RESPONSE_LINE_RE.finditer(text) if m.group(1) == status)
        pairs.append(
            QAPair(
                question=f"¿Qué significa la respuesta HTTP {status} del endpoint `{endpoint_label}`?",
                answer=desc.strip(),
                evidence_span=line_match.group(0),
                answer_type="extractive",
                generation_detail="openapi_operation_response",
            )
        )

    return pairs


def _qa_from_openapi_schema(text: str) -> list[QAPair]:
    """Genera preguntas sobre un chunk que empieza con "Schema: ..." (el tipo del
    schema, y el tipo/valores permitidos de cada propiedad), usando expresiones
    regulares sobre el formato fijo que escribe parse_openapi.py.
    """
    pairs: list[QAPair] = []

    header_match = _SCHEMA_HEADER_RE.search(text)
    if not header_match:
        return pairs
    schema_name = header_match.group(1).strip()

    type_match = _SCHEMA_TYPE_RE.search(text)
    if type_match:
        pairs.append(
            QAPair(
                question=f"¿Qué tipo de dato es el schema `{schema_name}`?",
                answer=type_match.group(1).strip(),
                evidence_span=type_match.group(0),
                answer_type="extractive",
                generation_detail="openapi_schema_type",
            )
        )

    for prop_name, prop_type, enum, desc in _SCHEMA_PROP_LINE_RE.findall(text):
        line_match = next(m for m in _SCHEMA_PROP_LINE_RE.finditer(text) if m.group(1) == prop_name)
        line = line_match.group(0)
        pairs.append(
            QAPair(
                question=f"¿Qué tipo de dato tiene el campo `{prop_name}` del schema `{schema_name}`?",
                answer=f"{prop_type}. {desc}".strip(".").strip() + ("." if desc else ""),
                evidence_span=line,
                answer_type="extractive",
                generation_detail="openapi_schema_property_type",
            )
        )
        if enum:
            pairs.append(
                QAPair(
                    question=f"¿Cuáles son los valores válidos para el campo `{prop_name}` del schema `{schema_name}`?",
                    answer=enum,
                    evidence_span=line,
                    answer_type="enum",
                    generation_detail="openapi_schema_property_enum",
                )
            )

    return pairs


# --- Plantillas para tablas de Markdown (content_type == "table") -------------------


def _strip_backticks(value: str) -> str:
    """Varias tablas de docs_raw ya envuelven un identificador en comillas
    invertidas dentro de la celda (por ejemplo '`collector_id`', '`GET /`'). Sin
    esto, envolverlo de nuevo en comillas invertidas al armar la pregunta
    generaria comillas dobles feas (` ``collector_id`` `) -- se quitan las que ya
    traia la fuente antes de aplicar el propio formato de la pregunta, una sola
    vez y de forma consistente."""
    return value.strip().strip("`").strip()


def _parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]] | None:
    """Un lector minimo de tablas estilo GitHub: separa cada fila por el
    caracter '|', y descarta la fila separadora (---|---). No maneja el caso de
    un '|' escapado dentro de una celda (no se encontraron casos asi en
    docs_raw) -- es una limitacion aceptada y documentada, no un intento de
    cubrir absolutamente cualquier variante de tabla en Markdown."""
    lines = [line for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return None

    def split_row(line: str) -> list[str]:
        cells = line.strip().strip("|").split("|")
        return [c.strip() for c in cells]

    header = split_row(lines[0])
    data_rows = [split_row(line) for line in lines[2:] if line.strip()]  # lines[1] es la fila separadora ---|---
    data_rows = [row for row in data_rows if len(row) == len(header)]  # descarta filas mal formadas en vez de fallar
    return header, data_rows


# Nombres de columna (pasados a minuscula) que se reconocen como "identificador de
# la fila" vs. "tipo de dato" vs. "codigo de error" -- estan basados en los
# encabezados reales que se encontraron en docs_raw (Field/Type en las tablas de
# diccionario de datos de 2p-revenue-optimizer-api; Code/Error/Description en las
# tablas de codigos de error de catalog-portfolio-api; Alert/Description en
# vendor-stockkeeper-api/alerts.md).
_ID_COLUMN_NAMES = {"field", "column", "parameter", "name", "alert"}
_TYPE_COLUMN_NAMES = {"type"}
_CODE_COLUMN_NAMES = {"code", "status"}
_DESC_COLUMN_NAMES = {"description", "error", "descripcion", "descripción"}


def _qa_from_table(text: str, section_label: str) -> list[QAPair]:
    """Genera 1 pregunta por cada fila de una tabla estilo GitHub: primero
    intenta reconocer un patron de columnas conocido (Field+Type, Code+Description),
    y si el encabezado no coincide con ninguno conocido, cae a un patron generico
    de "que informacion da la tabla sobre X" -- ver _ID_COLUMN_NAMES y las
    constantes vecinas arriba, basadas en encabezados reales de docs_raw.
    """
    parsed = _parse_markdown_table(text)
    if not parsed:
        return []
    header, rows = parsed
    header_lower = [h.lower() for h in header]

    id_idx = next((i for i, h in enumerate(header_lower) if h in _ID_COLUMN_NAMES), None)
    type_idx = next((i for i, h in enumerate(header_lower) if h in _TYPE_COLUMN_NAMES), None)
    code_idx = next((i for i, h in enumerate(header_lower) if h in _CODE_COLUMN_NAMES), None)
    desc_idx = next((i for i, h in enumerate(header_lower) if h in _DESC_COLUMN_NAMES), None)

    pairs: list[QAPair] = []
    for row in rows:
        # Reconstruye la linea ORIGINAL de la tabla para esta fila, para usarla
        # como evidence_span (tiene que ser una subcadena literal de chunk.text).
        row_line = "| " + " | ".join(row) + " |"
        if row_line not in text:
            # El formato original puede tener un espaciado distinto (columnas
            # alineadas con espacios de relleno) -- se busca la linea real dentro
            # del texto en vez de asumir que la reconstruccion coincide caracter
            # por caracter.
            candidate = next((line for line in text.splitlines() if all(cell in line for cell in row if cell)), None)
            if candidate is None:
                continue
            row_line = candidate

        if id_idx is not None and type_idx is not None and row[id_idx] and row[type_idx]:
            extra = f" ({row[desc_idx]})" if desc_idx is not None and row[desc_idx] else ""
            pairs.append(
                QAPair(
                    question=f"¿Qué tipo de dato tiene `{_strip_backticks(row[id_idx])}` en {section_label}?",
                    answer=f"{row[type_idx]}{extra}",
                    evidence_span=row_line,
                    answer_type="extractive",
                    generation_detail="table_field_type",
                )
            )
        elif code_idx is not None and desc_idx is not None and row[code_idx] and row[desc_idx]:
            pairs.append(
                QAPair(
                    question=f"¿Qué significa el código `{_strip_backticks(row[code_idx])}` en {section_label}?",
                    answer=row[desc_idx],
                    evidence_span=row_line,
                    answer_type="extractive",
                    generation_detail="table_code_description",
                )
            )
        elif id_idx is not None:
            otros = [f"{header[i]}: {v}" for i, v in enumerate(row) if i != id_idx and v]
            if otros:
                pairs.append(
                    QAPair(
                        question=f"¿Qué información documenta {section_label} sobre `{_strip_backticks(row[id_idx])}`?",
                        answer="; ".join(otros),
                        evidence_span=row_line,
                        answer_type="extractive",
                        generation_detail="table_generic_row",
                    )
                )

    return pairs


def generate_template_qa(text: str, content_type: str, section_path: str) -> list[QAPair]:
    """Genera pares de pregunta-respuesta deterministas para un chunk de tabla o
    de OpenAPI; devuelve una lista vacia para cualquier otro content_type (la
    prosa y el codigo pasan por qa_llm_generate.py en su lugar).

    Por que se decide segun content_type y no adivinando por el contenido del
    texto: content_type ya lo calculo con certeza chunking.py, asi que reusarlo
    evita tener que volver a detectar el tipo de fuente aca (ver
    notebooks/diagnostico.ipynb, seccion 9, sobre el enfoque hibrido de
    plantillas + modelo de lenguaje).
    """
    if content_type == "api_spec":
        if "Endpoint:" in text:
            return _qa_from_openapi_operation(text)
        if "Schema:" in text:
            return _qa_from_openapi_schema(text)
        return []
    if content_type == "table":
        section_label = f"la tabla de `{_strip_backticks(section_path.split(' > ')[-1])}`"
        return _qa_from_table(text, section_label)
    return []
