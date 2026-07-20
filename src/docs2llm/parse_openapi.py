"""Lee un archivo specs/swagger.yaml (una especificacion OpenAPI 3.0.x) y lo
convierte en Secciones: una por operacion (ruta + metodo HTTP) y una por schema
(la forma de un objeto de datos), nunca se trata como si fuera texto Markdown
generico.

Por que un lector dedicado en vez de tratar el YAML como texto plano: un
swagger.yaml tiene estructura rica (parametros, valores permitidos cerrados,
codigos de respuesta, requisitos de autenticacion) que se pierde si se corta por
lineas -- partir un endpoint a la mitad, entre "parameters:" y "responses:",
genera un fragmento que no sirve ni para busqueda ni para hacer una pregunta con
respuesta completa. Ademas, se confirmo que los 3 archivos swagger.yaml de
docs_raw (price-engine-api, traffic-gate-api, zenith-keeper) usan convenciones
distintas entre si (price-engine-api usa un `$ref` fuera de lo estandar, que
apunta a `#/models/...` en vez de `#/components/schemas/...`). Por eso, la
funcion que sigue esos punteros (`$ref`) camina el diccionario YA LEIDO siguiendo
la ruta literal, sin asumir que siempre va a empezar en "components/" -- funciona
igual para las 2 convenciones.

Cada operacion y cada schema se convierte en una Section con un UNICO Block de
tipo "api_spec": a diferencia de las secciones de Markdown, estas unidades ya salen 
del tamaño correcto para ser un chunk por si solas -- chunking.py las trata como "ya listas".
"""

from __future__ import annotations

from typing import Any

# PyYAML lee el archivo swagger.yaml y lo convierte en diccionarios y listas de Python, de la misma
# forma que se usa en config.py para leer config/config.yaml.
import yaml

from docs2llm.parse_units import Block, Section

_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}


def _resolve_ref(spec: dict, ref: str) -> Any:
    """Sigue un puntero `$ref` (por ejemplo '#/components/schemas/Block' o
    '#/models/Zones') caminando el diccionario ya leido, un segmento a la vez.
    Devuelve None si esa ruta no existe (en vez de lanzar un error): un
    `$ref` roto en el YAML no deberia tumbar todo el pipeline, solo esa referencia
    puntual queda sin resolver y se muestra tal cual esta (ver _describe_schema_ref).
    """
    if not ref.startswith("#/"):
        return None  # no se soportan referencias a archivos externos, no aparecen en docs_raw
    node: Any = spec
    for segment in ref[2:].split("/"):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def _ref_name(ref: str) -> str:
    """El ultimo segmento de un `$ref`, para mostrarlo compacto: '#/models/Zones' -> 'Zones'."""
    return ref.rstrip("/").rsplit("/", 1)[-1]


def _describe_schema_ref(spec: dict, schema_obj: dict, indent: str = "") -> list[str]:
    """Describe un schema (ya sea escrito directo ahi, o referenciado con `$ref`)
    en 1-2 lineas de texto plano, sin bajar en profundidad de forma recursiva --
    para un requestBody o response se prioriza que el resultado sea legible y
    compacto, por encima de detallar cada nivel de anidamiento.
    """
    if "$ref" in schema_obj:
        name = _ref_name(schema_obj["$ref"])
        return [f"{indent}-> schema '{name}' (ver chunk de schemas para el detalle completo)"]

    schema_type = schema_obj.get("type", "object")
    lines = [f"{indent}type: {schema_type}"]

    if schema_type == "array" and "items" in schema_obj:
        lines += _describe_schema_ref(spec, schema_obj["items"], indent=indent + "  ")
    elif "properties" in schema_obj:
        for prop_name, prop_schema in schema_obj["properties"].items():
            prop_type = prop_schema.get("type", "$ref" in prop_schema and _ref_name(prop_schema["$ref"]) or "object")
            enum = prop_schema.get("enum")
            enum_txt = f" enum={enum}" if enum else ""
            lines.append(f"{indent}  - {prop_name} ({prop_type}){enum_txt}")
    elif "enum" in schema_obj:
        lines.append(f"{indent}  enum={schema_obj['enum']}")

    return lines


def _format_operation(spec: dict, path: str, method: str, operation: dict) -> str:
    """Arma un texto compacto y legible para una operacion OpenAPI (ruta + metodo HTTP).

    Por que el formato exacto importa: qa_templates.py despues vuelve a leer este
    texto con expresiones regulares dirigidas especificamente a este formato (ver
    notebooks/diagnostico.ipynb, seccion 9) -- si el formato cambia aca, esas
    expresiones regulares se tienen que actualizar junto con el.
    """
    lines = [f"Endpoint: {method.upper()} {path}"]

    if operation.get("operationId"):
        lines.append(f"Operation ID: {operation['operationId']}")
    if operation.get("summary"):
        lines.append(f"Summary: {operation['summary']}")
    if operation.get("description"):
        lines.append(f"Description: {operation['description']}")

    parameters = operation.get("parameters", [])
    if parameters:
        lines.append("Parameters:")
        for raw_param in parameters:
            param = _resolve_ref(spec, raw_param["$ref"]) if "$ref" in raw_param else raw_param
            if param is None:
                continue  # $ref roto: se omite en vez de reventar el parseo de todo el spec
            name = param.get("name", "?")
            location = param.get("in", "?")
            required = " [required]" if param.get("required") else ""
            desc = param.get("description", "")
            schema = param.get("schema", {})
            type_ = schema.get("type", "?")
            enum = schema.get("enum")
            enum_txt = f" enum={enum}" if enum else ""
            lines.append(f"  - {name} (in: {location}, type: {type_}){enum_txt}{required}: {desc}")

    request_body = operation.get("requestBody")
    if request_body:
        required = " [required]" if request_body.get("required") else ""
        lines.append(f"Request body{required}:")
        if request_body.get("description"):
            lines.append(f"  {request_body['description']}")
        for media_type, media_obj in request_body.get("content", {}).items():
            lines.append(f"  content-type: {media_type}")
            lines += _describe_schema_ref(spec, media_obj.get("schema", {}), indent="  ")

    responses = operation.get("responses", {})
    if responses:
        lines.append("Responses:")
        for status, raw_resp in responses.items():
            resp = _resolve_ref(spec, raw_resp["$ref"]) if "$ref" in raw_resp else raw_resp
            if resp is None:
                continue
            desc = resp.get("description", "")
            lines.append(f"  - {status}: {desc}")

    if operation.get("security") is not None:
        lines.append(f"Security: {operation['security']}")

    return "\n".join(lines)


def _format_schema(name: str, schema: dict) -> str:
    """Arma un texto compacto para una entrada de components/schemas.
    Depende del mismo formato fijo que _format_operation, y qa_templates.py
    necesita este formato exacto para poder extraer preguntas con expresiones regulares.
    """
    lines = [f"Schema: {name}", f"Type: {schema.get('type', 'object')}"]

    if schema.get("type") == "array" and "items" in schema:
        item_type = schema["items"].get("type", "object")
        item_desc = schema["items"].get("description", "")
        lines.append(f"Items: {item_type}" + (f" ({item_desc})" if item_desc else ""))

    properties = schema.get("properties", {})
    if properties:
        lines.append("Properties:")
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type") or ("$ref" in prop_schema and f"-> {_ref_name(prop_schema['$ref'])}") or "object"
            desc = prop_schema.get("description", "")
            enum = prop_schema.get("enum")
            enum_txt = f" enum={enum}" if enum else ""
            lines.append(f"  - {prop_name} ({prop_type}){enum_txt}: {desc}")

    required = schema.get("required")
    if required:
        lines.append(f"Required: {required}")

    return "\n".join(lines)


def parse_openapi_sections(text: str) -> list[Section]:
    """Lee un swagger.yaml y devuelve una Section por cada operacion (ruta+metodo)
    y una por cada schema.

    Por que char_start/char_end quedan siempre en 0: a diferencia de un archivo
    .md, el texto de cada chunk aca es GENERADO por este mismo modulo (ver
    _format_operation/_format_schema), no una porcion literal recortada del
    archivo original -- no existe una posicion de caracter real para rastrear. La
    procedencia del dato sigue intacta igual, a traves de source_path + section_path.
    """
    spec = yaml.safe_load(text)
    sections: list[Section] = []

    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(operation, dict):
                continue  # ignora claves como 'parameters' compartidos a nivel de ruta, que no son un metodo HTTP
            block_text = _format_operation(spec, path, method, operation)
            sections.append(
                Section(
                    heading=f"{method.upper()} {path}",
                    level=1,
                    section_path=("swagger.yaml", "paths", f"{method.upper()} {path}"),
                    blocks=[Block(content_type="api_spec", text=block_text, char_start=0, char_end=len(block_text))],
                )
            )

    schema_sources: dict[str, dict] = {}
    schema_sources.update((spec.get("components") or {}).get("schemas") or {})
    schema_sources.update(spec.get("models") or {})

    for name, schema in schema_sources.items():
        if not isinstance(schema, dict):
            continue
        block_text = _format_schema(name, schema)
        sections.append(
            Section(
                heading=name,
                level=1,
                section_path=("swagger.yaml", "schemas", name),
                blocks=[Block(content_type="api_spec", text=block_text, char_start=0, char_end=len(block_text))],
            )
        )

    n_operaciones = sum(1 for s in sections if s.section_path[1:2] == ("paths",))
    n_schemas = sum(1 for s in sections if s.section_path[1:2] == ("schemas",))
    print(f"  [parse_openapi] {n_operaciones} operaciones, {n_schemas} schemas")
    return sections
