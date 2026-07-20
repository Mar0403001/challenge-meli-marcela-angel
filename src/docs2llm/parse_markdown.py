"""Convierte un archivo Markdown ya limpio en un arbol de secciones, organizado por
titulo (heading).

La idea central (ver notebooks/diagnostico.ipynb, seccion 2) es que, en vez de
partir el archivo en trozos de N caracteres sin mirar su contenido, se usa el AST
real de Markdown -- el arbol de sintaxis que arma la libreria markdown-it-py -- para
saber donde empieza y termina cada seccion (H1/H2/H3/...), y dentro de cada
seccion, que partes son "atomicas" (tablas, bloques de codigo, listas, HTML) y por
lo tanto nunca se deben cortar a la mitad.

Antes de escribir este modulo se comprobo como se comporta
markdown-it-py:
  - Con el modo 'commonmark' + `.enable('table')`, las tablas estilo GitHub se
    convierten en un unico "token" `table_open`, que ya sabe donde empieza y
    termina toda la tabla (encabezado y filas).
  - Un bloque ```lenguaje ... ``` se convierte en un unico token `fence`, y una
    linea que empieza con `#` DENTRO de ese bloque (por ejemplo un comentario de
    bash) NO se interpreta como un titulo -- confirma que no hace falta escribir
    logica propia para evitar esa confusion.
  - Un bloque de HTML como `<div>...</div>` o `<details>...</details>` se convierte
    en un unico token `html_block` que cubre todo el bloque, aunque tenga etiquetas
    anidadas adentro.
  - Todo token de "nivel superior" (titulo, tabla, bloque de codigo, HTML, parrafo,
    lista, cita) aparece marcado con `token.level == 0`; lo que hay dentro de el
    (por ejemplo las celdas de una tabla, o los items de una lista) tiene un nivel
    mayor y se puede ignorar sin perder nada, porque el token de afuera ya resume
    todo su rango de lineas.
"""

from __future__ import annotations

import re

# markdown-it-py es la libreria que lee un archivo Markdown y lo convierte en una
# lista de "tokens" (titulos, parrafos, tablas, bloques de codigo...) en vez de
# dejarlo como texto plano suelto -- eso es lo que permite saber, con certeza, donde
# empieza y termina cada parte del documento.
from markdown_it import MarkdownIt

from docs2llm.parse_html import html_block_to_text
from docs2llm.parse_units import Block, Section

# Quita **negrita**/__negrita__ de los TITULOS de encabezado:
# se encontro que en demand-forecast-docs/overview.md varios titulos usan negrita de
# Markdown (por ejemplo "## **Git**"), y sin este arreglo section_path/heading
# terminarian mostrando los asteriscos tal cual, en vez de un titulo limpio. Solo se
# quita la negrita marcada con doble simbolo (** o __).
_BOLD_MARKUP_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")


def _clean_heading_title(raw_title: str) -> str:
    return _BOLD_MARKUP_RE.sub(lambda m: m.group(1) or m.group(2), raw_title).strip()

# Se arranca del modo 'commonmark' (el mas estricto y predecible de markdown-it-py)
# y se habilita SOLO el soporte de tablas ('table').
_MD = MarkdownIt("commonmark").enable("table")

# Estos son los tipos de "token" (de nivel 0) que representan un bloque de contenido
# "atomico": nunca se deben partir a la mitad al armar los chunks mas adelante (ver
# chunking.py). Aca solo se decide QUE es atomico y de que tipo; la decision de como
# empaquetar esos bloques en chunks vive en otro modulo.
_ATOMIC_CODE_TYPES = {"fence"}
_ATOMIC_TABLE_TYPES = {"table_open"}
# Estos se tratan como un unico bloque de prosa, sin separarlos por item de lista o
# por linea de cita: partir una lista de pasos (por ejemplo, de troubleshooting) a
# la mitad entre dos chunks distintos seria peor que dejarla completa aunque quede
# un poco mas larga.
_ATOMIC_PROSE_CONTAINER_TYPES = {"html_block", "paragraph_open", "bullet_list_open", "ordered_list_open", "blockquote_open"}

_ALL_BLOCK_OPEN_TYPES = _ATOMIC_CODE_TYPES | _ATOMIC_TABLE_TYPES | _ATOMIC_PROSE_CONTAINER_TYPES


# Block y Section viven en parse_units.py (compartidas con parse_openapi.py) --
# ver ese modulo para saber por que las dos comparten la misma estructura.


def _line_start_offsets(text: str) -> list[int]:
    """Calcula en que caracter empieza cada linea del texto (el indice de la lista
    es el numero de linea, empezando en 0).

    markdown-it-py da los rangos de cada bloque en numero de linea (por ejemplo
    [4, 7)), no en posicion de caracter. Esta tabla se calcula una sola vez por
    archivo, para poder convertir "de la linea 4 a la 7" en "del caracter 812 al
    903" sin tener que recorrer el texto de nuevo por cada bloque.
    """
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def parse_markdown_sections(text: str, *, fallback_title: str) -> list[Section]:
    """Arma el arbol de secciones (titulo -> sus bloques) de un Markdown ya limpio.

    Por que un arbol y no una lista plana: es lo que le permite despues a
    chunking.py armar chunks respetando los titulos, sin partir tablas ni bloques
    de codigo a la mitad (ver notebooks/diagnostico.ipynb, seccion 2).

    `fallback_title`: un titulo "de emergencia" para el contenido que aparece antes
    del primer titulo real, o para archivos que no tienen ningun titulo -- asegura
    que section_path nunca quede vacio.
    """
    lines = text.splitlines()
    line_offsets = _line_start_offsets(text)

    def slice_lines(start_line: int, end_line: int) -> str:
        # end_line no se incluye, tal como lo entrega markdown-it-py en `.map`.
        return "\n".join(lines[start_line:end_line])

    tokens = _MD.parse(text)

    sections: list[Section] = []
    # Pila de (nivel, titulo) de los titulos actualmente "abiertos" -- define la
    # ruta del contenido que se esta leyendo en este punto del archivo.
    heading_stack: list[tuple[int, str]] = []

    # Seccion "de emergencia" para cualquier contenido que aparezca antes del
    # primer titulo real (o para archivos sin ningun titulo). Se descarta al final
    # si termino sin contenido.
    preamble = Section(heading=None, level=0, section_path=(fallback_title,), blocks=[])
    current_section = preamble
    sections.append(preamble)

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # Lo que esta DENTRO de un contenedor (celdas de una tabla, items de una
        # lista, texto en linea) tiene nivel > 0 y ya esta resumido por el .map del
        # token de afuera -- se ignora para no procesar el mismo contenido dos veces.
        if token.level > 0:
            i += 1
            continue

        if token.type == "heading_open":
            level = int(token.tag[1])  # "h1" -> 1, "h2" -> 2, ...
            title = _clean_heading_title(tokens[i + 1].content)  # el token 'inline' siempre viene justo despues

            # Cierra cualquier titulo de nivel igual o mayor: un
            # nuevo H2 termina la seccion del H2 anterior (mismo nivel) y de
            # cualquier H3/H4 que hubiera quedado abierto por debajo (nivel mayor =
            # mas profundo, mas anidado).
            heading_stack = [(lvl, t) for lvl, t in heading_stack if lvl < level]
            heading_stack.append((level, title))

            section_path = tuple(t for _, t in heading_stack)
            current_section = Section(heading=title, level=level, section_path=section_path, blocks=[])
            sections.append(current_section)
            i += 2  # se salta el token 'inline' del titulo, que ya se leyo arriba
            continue

        if token.type in _ALL_BLOCK_OPEN_TYPES and token.map is not None:
            start_line, end_line = token.map
            raw_text = slice_lines(start_line, end_line)

            if token.type in _ATOMIC_CODE_TYPES:
                content_type = "code"
                block_text = raw_text
            elif token.type in _ATOMIC_TABLE_TYPES:
                content_type = "table"
                block_text = raw_text
            else:
                content_type = "prose"
                # Solo el HTML necesita limpieza; los parrafos/listas/citas de
                # Markdown ya son texto legible tal cual estan en el archivo fuente.
                block_text = html_block_to_text(raw_text) if token.type == "html_block" else raw_text

            block_text = block_text.strip()
            if block_text:  # una tabla/parrafo vacio no aporta nada, se descarta
                char_start = line_offsets[start_line]
                current_section.blocks.append(
                    Block(content_type=content_type, text=block_text, char_start=char_start, char_end=char_start + len(block_text))
                )

        i += 1

    # La seccion "de emergencia" (preambulo) solo se conserva si termino con
    # contenido real (la gran mayoria de archivos arrancan directo con un titulo
    # real, asi que en el caso comun esta seccion queda vacia y se descarta aca,
    # sin propagarse como ruido).
    resultado = [s for s in sections if s.blocks]
    n_blocks = sum(len(s.blocks) for s in resultado)
    print(f"  [parse_markdown] {len(resultado)} secciones, {n_blocks} bloques")
    return resultado
