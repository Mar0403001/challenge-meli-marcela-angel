"""Segundo paso del pipeline: limpia el texto crudo de un archivo Markdown ANTES de
convertirlo en un arbol de secciones con markdown-it-py (ver parse_markdown.py).

Cada funcion de este archivo arregla un problema puntual que se encontro y se
confirmo en docs_raw/ -- no es una regla generica "por si las dudas". Ver
notebooks/diagnostico.ipynb, seccion 3, para la evidencia completa de cada uno.
"""

from __future__ import annotations

import html
import re

# --- 1. Titulos mal escritos (falta el espacio despues de '#') ----------------------
#
# zenith-keeper tiene titulos como "#Use cases" y "#API FRESH PRODUCTS DOCUMENTATION".
# Segun las reglas de Markdown, un '#' pegado al texto (sin espacio) NO cuenta como
# titulo -- es un parrafo normal que arranca con el caracter '#'. Si esto no se
# corrige antes de leer el archivo, esas dos secciones quedarian invisibles para el
# arbol de secciones (todo su contenido caeria dentro del "preambulo" con el nombre
# del archivo, perdiendo el titulo real).
#
# Esto solo se aplica FUERA de los bloques de codigo: un "#comentario" dentro de un
# bloque bash (por ejemplo payment-promise-gateway/guide/README.md tiene
# "# Common variables" dentro de un bloque ```bash) no debe tocarse -- eso es codigo,
# no un titulo mal escrito.
_FENCE_MARKER_RE = re.compile(r"^(```+|~~~+)")
_BROKEN_ATX_RE = re.compile(r"^(#{1,6})([^#\s])")


def fix_broken_atx_headings(text: str) -> str:
    """Agrega el espacio que falta en titulos mal escritos (`#Titulo` -> `# Titulo`),
    pero solo fuera de los bloques de codigo.

    Por que hace falta esto: las reglas de Markdown no reconocen un '#' pegado al
    texto como un titulo -- sin este arreglo, esas secciones quedarian invisibles
    para el arbol de secciones (ver notebooks/diagnostico.ipynb, seccion 3).
    """
    out_lines: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if _FENCE_MARKER_RE.match(line.strip()):
            # No hace falta llevar la cuenta exacta de comillas invertidas vs.
            # virgulillas, ni cuantos caracteres tiene cada delimitador: se
            # confirmo que docs_raw solo usa bloques simples de triple comilla
            # invertida, sin anidar uno dentro de otro, asi que alcanza con
            # "prender/apagar" un interruptor cada vez que se ve uno.
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if not in_fence:
            # Reemplaza SOLO el "#" pegado al primer caracter (ej. "#U" -> "# U");
            # el resto de la linea ("se cases") no se toca, porque no forma parte
            # de lo que la expresion regular captura.
            line = _BROKEN_ATX_RE.sub(lambda m: f"{m.group(1)} {m.group(2)}", line)
        out_lines.append(line)
    return "\n".join(out_lines)


# --- 2. Comillas y simbolos convertidos a codigo HTML dentro de bloques de codigo ----
#
# vendor-stockkeeper-api/0.0.10-stock-consumer (la carpeta que se eligio como version
# canonica, ver config/config.yaml, clave canonical_sources) tiene comillas y los
# simbolos '<'/'>' convertidos a su version "escapada" en HTML (&#34;, &#39;, &gt;,
# &lt;) DENTRO de bloques de codigo JSON -- se conto: 546 apariciones de &#34; en
# important.md, contra 0 en la misma seccion de otras versiones del mismo proyecto.
# Es un bug de esa exportacion puntual (probablemente se genero a partir de una
# pagina HTML ya renderizada), no un uso a proposito de esos codigos.
#
# Se aplica a CUALQUIER archivo, no solo a ese proyecto, porque la operacion no hace
# nada donde el bug no existe: si un bloque de codigo no tiene simbolos escapados,
# html.unescape() (una funcion de la libreria estandar de Python) no cambia nada. Es
# mas simple y mas seguro que escribir a mano un caso especial tipo
# "si el proyecto es vendor-stockkeeper-api, hacer esto".
_CODE_FENCE_BLOCK_RE = re.compile(r"(^```[^\n]*\n)(.*?)(\n```)", re.DOTALL | re.MULTILINE)


def unescape_html_entities_in_code_fences(text: str) -> str:
    """Revierte comillas/`<`/`>` que quedaron convertidas a codigo HTML (`&#34;`,
    `&gt;`...) dentro de bloques de codigo.

    Por que se aplica a todo archivo y no solo a un proyecto puntual: como no hace
    nada donde el bug no existe, es mas simple y mas seguro que escribir un caso
    especial por proyecto (ver notebooks/diagnostico.ipynb, seccion 3, el bug de
    vendor-stockkeeper-api).
    """

    def _unescape_body(match: re.Match) -> str:
        opening, body, closing = match.groups()
        return opening + html.unescape(body) + closing

    return _CODE_FENCE_BLOCK_RE.sub(_unescape_body, text)


# --- 3. Lineas de estilo repetidas dentro del MISMO diagrama Mermaid ----------------
#
# (Mermaid es una forma de escribir diagramas como texto, dentro de un bloque de
# codigo.) demand-forecast-docs/overview.md tiene 2 diagramas grandes (entre 250 y
# 280 lineas cada uno) donde una misma linea de estilo "style <nodo> fill:..."
# aparece 2 veces dentro del mismo diagrama (por ejemplo "style step__train fill:
# #fff" aparece en las lineas 401 y 520 del mismo bloque, a 120 lineas de distancia
# -- no son lineas pegadas por error, es la misma instruccion repetida en dos
# puntos del diagrama). Esto es ruido DENTRO de un mismo bloque de codigo, distinto
# a un contenido duplicado ENTRE dos fragmentos separados del corpus -- por eso se
# arregla aca y no en dedup.py (ver notebooks/diagnostico.ipynb, seccion 3, para la
# diferencia completa). Se deja solo la primera aparicion de cada linea de estilo
# exacta; el resto del diagrama (nodos, flechas, subgrupos) no se toca.
_MERMAID_BLOCK_RE = re.compile(r"(```mermaid\n)(.*?)(\n```)", re.DOTALL)


def collapse_duplicate_mermaid_style_lines(text: str) -> str:
    """Elimina lineas `style <nodo> ...` repetidas dentro del MISMO diagrama Mermaid.

    Por que se arregla aca y no en dedup.py: es ruido dentro de una sola unidad de
    contenido (un mismo bloque de codigo), no un duplicado entre dos fragmentos
    distintos -- la diferencia esta explicada en notebooks/diagnostico.ipynb, seccion 3.
    """

    def _dedupe_styles(match: re.Match) -> str:
        opening, body, closing = match.groups()
        seen_style_lines: set[str] = set()
        kept_lines: list[str] = []
        for line in body.split("\n"):
            if line.strip().startswith("style "):
                if line in seen_style_lines:
                    continue  # misma directiva ya vista antes en este mismo diagrama
                seen_style_lines.add(line)
            kept_lines.append(line)
        return opening + "\n".join(kept_lines) + closing

    return _MERMAID_BLOCK_RE.sub(_dedupe_styles, text)


def normalize_markdown_text(text: str) -> str:
    """Aplica los 3 fixes de este modulo, en orden, sobre un archivo .md crudo.

    El orden importa: fix_broken_atx_headings corre primero para que, si algun
    encabezado roto estuviera dentro de lo que mas tarde se detecta como fence (no
    pasa en este corpus, pero es la relacion segura), el resto de pasos ya trabajen
    sobre fences bien delimitados.
    """
    antes = text
    text = fix_broken_atx_headings(text)
    if text != antes:
        print("  [normalize] encabezados ATX rotos corregidos")

    antes = text
    text = unescape_html_entities_in_code_fences(text)
    if text != antes:
        print("  [normalize] entidades HTML doble-escapadas revertidas dentro de bloques de codigo")

    antes = text
    text = collapse_duplicate_mermaid_style_lines(text)
    if text != antes:
        print("  [normalize] lineas de estilo Mermaid repetidas colapsadas")

    return text
