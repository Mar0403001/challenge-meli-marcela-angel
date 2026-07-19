"""Estas dos clases las comparten parse_markdown.py y parse_openapi.py.

Los dos parsers (el de Markdown y el de OpenAPI) devuelven el MISMO tipo de árbol
(una lista de Section, cada una con sus Block adentro), aunque lean fuentes muy
distintas (texto Markdown vs. un archivo YAML de especificación). Gracias a esto,
chunking.py puede tratar cualquiera de los dos tipos de archivo de la misma forma,
sin tener que escribir un `if file_kind == "markdown" ... else ...` repartido por
todo su código.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Block:
    """La unidad más chica de contenido dentro de una sección: una tabla completa,
    un bloque de código, o un párrafo de prosa. Nunca se parte a la mitad."""

    content_type: str  # "prose" | "table" | "code" | "api_spec" -- ver corpus.jsonl
    text: str
    char_start: int
    char_end: int  # == char_start + len(text), por construccion


@dataclasses.dataclass
class Section:
    """Una sección del documento: un título (heading) y todo el contenido que cae
    debajo de él, hasta el próximo título del mismo nivel o superior."""

    heading: str | None  # None solo para el preambulo sintetico de un .md
    level: int
    section_path: tuple[str, ...]
    blocks: list[Block]
