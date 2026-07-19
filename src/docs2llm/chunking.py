"""Tercer paso del pipeline: convierte la lista de Section (que vino de
parse_markdown.py o de parse_openapi.py) en la lista final de Chunk que se
escribe en corpus.jsonl.

Reglas de armado (bin-packing), en orden de prioridad:

  1. Un bloque atomico (una tabla, un bloque de codigo, una especificacion de API)
     NUNCA se parte a la mitad entre dos chunks -- se guarda entero en uno o en otro.
  2. El "presupuesto" de un chunk es blando hacia arriba (target_tokens) y duro como
     tope (max_tokens): se van sumando bloques seguidos de la MISMA seccion hasta
     acercarse al objetivo; si agregar el siguiente bloque pasaria el maximo, se
     cierra el chunk actual y se abre uno nuevo.
  3. EXCEPCION a la regla 1: si un UNICO bloque atomico, por si solo, ya supera
     max_tokens (un caso raro: no se encontro ningun caso real en docs_raw al
     escribir esto, pero el codigo lo cubre por las dudas), se usa una ventana con
     solapamiento (overlap_tokens) -- el UNICO lugar de todo el pipeline donde de
     verdad se corta texto a la mitad.
  4. Secciones de Markdown muy chicas (con menos de target_tokens/4) NO se emiten
     como su propio chunk: sus bloques se "cargan hacia adelante" y se juntan con
     los bloques de la SIGUIENTE seccion del mismo documento (una aproximacion a
     propósito a "juntar con la seccion hermana": reconstruir el arbol exacto de
     padres/hermanos no se justificaba para el tiempo disponible, y la siguiente
     seccion en el orden real del documento casi siempre es su contexto natural,
     ver notebooks/diagnostico.ipynb, seccion 5).
  5. Las secciones que vienen de parse_openapi.py (content_type == "api_spec") NUNCA
     se juntan con sus vecinas aunque sean chicas: un schema corto como "Zones" (3
     lineas) sigue siendo una unidad completa y util por si sola para busqueda --
     juntarlo con el siguiente endpoint, que no tiene relacion, rompería el nivel
     de detalle que el lector de OpenAPI fue disenado a proposito para conservar.
  6. Un chunk que, incluso despues de intentar juntarse con otro, sigue siendo muy
     corto o es puro texto de relleno ("TO DO.", "TBD", etc.) NO se descarta en
     silencio: se conserva, pero se marca `is_low_signal=True`, para que la
     generacion de preguntas pueda decidir explicitamente no generar nada a partir
     de el (ver qa_templates.py y qa_llm_generate.py).

Conteo de tokens: se usa un tokenizador real, `tiktoken`, con el encoding
`cl100k_base` (el mismo que usan los modelos GPT-4/GPT-3.5, suficiente para medir 
el tamaño de los chunks). 
"""

from __future__ import annotations

import dataclasses
import functools
import re

import tiktoken

from docs2llm.config import ChunkingConfig
from docs2llm.dedup import compute_content_hash
from docs2llm.parse_units import Block

# cl100k_base es un encoding fijo (OpenAI no lo cambia para modelos que ya salieron
# al mercado), asi que guardar el Encoding en memoria una sola vez por proceso es
# seguro: no existe una version mas nueva que se este perdiendo por no recargarlo.
@functools.lru_cache(maxsize=1)
def _encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")

# Un chunk final con menos tokens que este numero se marca is_low_signal, sin
# importar si es el resultado de juntar varias secciones chicas -- es el umbral de
# "hay tan poco texto que ninguna pregunta seria podria salir de aca", que es
# distinto del umbral que decide CUANDO intentar juntar secciones (ver
# min_tokens_before_merge, mas abajo, dentro de chunk_sections).
_MIN_INFORMATIVE_TOKENS = 12

# Coincide (sin importar mayusculas/minusculas, ignorando puntuacion al final) con
# textos de relleno reales que se encontraron en docs_raw: "TO DO." (aparece 3
# veces en price-engine-api/general.md), "_TODO_" (en
# order-workflow-api/services/database.md). No es una lista de todo texto de
# relleno posible en cualquier documentacion -- es la lista de lo que
# efectivamente se encontro en este corpus.
_PLACEHOLDER_RE = re.compile(r"^_?(to\s*do|tbd|todo|pendiente|n/a)_?\.?$", re.IGNORECASE)


def _looks_like_placeholder(text: str) -> bool:
    """True si el texto ES (no solo si contiene) un texto de relleno conocido
    ("TO DO.", "_TODO_", etc.).

    Es una lista confirmada contra docs_raw real, no una lista generica -- ver
    _PLACEHOLDER_RE arriba.
    """
    return bool(_PLACEHOLDER_RE.match(text.strip()))


@dataclasses.dataclass
class Chunk:
    section_path: tuple[str, ...]
    chunk_index: int  # se asigna recien al final, cuando ya se sabe el total de chunks del documento
    n_chunks_in_doc: int
    text: str
    n_tokens: int  # conteo real, calculado con tiktoken (cl100k_base) -- ver el inicio de este archivo
    content_type: str  # "prose" | "table" | "code" | "api_spec" -- ver _dominant_content_type
    char_start: int
    char_end: int
    is_low_signal: bool


def _dominant_content_type(blocks: list[Block]) -> str:
    """Si un chunk mezcla tipos de contenido (por ejemplo prosa + tabla + prosa,
    algo comun en secciones de referencia de API), se etiqueta con el tipo del
    bloque mas grande en cantidad de caracteres -- una regla simple y siempre
    igual: el bloque mas grande es, en la practica, el que tiene la informacion
    central del chunk (la tabla o el codigo), y la prosa alrededor es contexto o
    explicacion de ese bloque, no al reves.
    """
    return max(blocks, key=lambda b: len(b.text)).content_type


def _count_tokens(text: str) -> int:
    return max(1, len(_encoder().encode(text)))


def _window_fallback(blocks: list[Block], config: ChunkingConfig) -> list[tuple[str, int, int]]:
    """El unico lugar de todo el pipeline que corta texto a la mitad: se usa SOLO
    cuando un bloque atomico, por si solo, ya supera max_tokens. Trabaja
    directamente sobre los identificadores que da tiktoken (no sobre posiciones de
    caracteres): max_tokens/overlap_tokens ya son cantidades de tokens reales, sin
    necesitar conversion. Devuelve (texto, cantidad_de_tokens, posicion_relativa)
    por cada ventana.
    """
    full_text = "\n\n".join(b.text for b in blocks)
    ids = _encoder().encode(full_text)
    step = max(1, config.max_tokens - config.overlap_tokens)

    windows: list[tuple[str, int, int]] = []
    start = 0
    while start < len(ids):
        window_ids = ids[start : start + config.max_tokens]
        windows.append((_encoder().decode(window_ids), len(window_ids), start))
        if start + config.max_tokens >= len(ids):
            break
        start += step
    return windows


def _pack_blocks(
    section_path: tuple[str, ...],
    blocks: list[Block],
    config: ChunkingConfig,
    duplicate_block_hashes: frozenset[str] = frozenset(),
) -> list[Chunk]:
    """Arma 1 o mas chunks a partir de una lista PLANA de bloques (ya en el orden
    en que se leen).

    Esta funcion no sabe nada de "secciones chicas que se juntan con otras" -- esa
    decision ya se tomo ANTES de llamarla (ver chunk_sections). Aca solo se decide
    cuantos chunks hacen falta para estos bloques puntuales, respetando
    target_tokens/max_tokens.

    `duplicate_block_hashes`: los hashes (ver dedup.compute_content_hash) de
    bloques que aparecen identicos en MAS DE UN archivo del corpus (por ejemplo, el
    boton HTML roto "Download X Template" que se repite en 5 archivos de
    catalog-portfolio-api, o el ejemplo de error JSON "brand_not_found" que se
    repite en 2 secciones distintas). Un bloque cuyo hash esta en este conjunto
    NUNCA se junta con sus vecinos -- se emite como su propio chunk, aislado. Sin
    esto, el mismo fragmento repetido quedaria mezclado con texto DISTINTO
    alrededor en cada archivo, y el chunk final de cada aparicion terminaria
    siendo distinto -- la deduplicacion por fila (dedup.py) jamas los detectaria
    como duplicados, aunque el fragmento de adentro sí se repita de verdad (esto
    se confirmo con un caso real, ver notebooks/diagnostico.ipynb, seccion 6, antes
    de agregar este mecanismo).
    """
    if not blocks:
        return []

    block_tokens = [_count_tokens(b.text) for b in blocks]

    # Caso excepcional (regla 3 de la explicacion al principio del archivo): un
    # unico bloque ya supera el maximo por si solo. No se encontro ningun caso real
    # en docs_raw, pero el codigo lo cubre de todas formas (ver
    # tests/test_chunking.py, que sí fuerza este caso a proposito).
    if len(blocks) == 1 and block_tokens[0] > config.max_tokens:
        chunks = []
        for window_text, n_tok, _rel_offset in _window_fallback(blocks, config):
            chunks.append(
                Chunk(
                    section_path=section_path,
                    chunk_index=-1,  # se asigna despues, en chunk_sections()
                    n_chunks_in_doc=-1,
                    text=window_text,
                    n_tokens=n_tok,
                    content_type=blocks[0].content_type,
                    char_start=blocks[0].char_start,
                    char_end=blocks[0].char_end,
                    is_low_signal=False,  # un bloque tan grande no cuenta como "poco informativo"
                )
            )
        return chunks

    chunks: list[Chunk] = []
    current: list[Block] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        text = "\n\n".join(b.text for b in current)
        n_tok = sum(_count_tokens(b.text) for b in current)
        chunks.append(
            Chunk(
                section_path=section_path,
                chunk_index=-1,
                n_chunks_in_doc=-1,
                text=text,
                n_tokens=n_tok,
                content_type=_dominant_content_type(current),
                char_start=min(b.char_start for b in current),
                char_end=max(b.char_end for b in current),
                is_low_signal=(n_tok < _MIN_INFORMATIVE_TOKENS) or (len(current) == 1 and _looks_like_placeholder(current[0].text)),
            )
        )

    for block, n_tok in zip(blocks, block_tokens):
        if duplicate_block_hashes and compute_content_hash(block.text) in duplicate_block_hashes:
            # Este bloque se detecto como repetido en OTRO archivo del corpus (ver
            # la explicacion de esta funcion, arriba): se aisla en su propio chunk,
            # nunca se mezcla con los bloques vecinos. Se marca is_low_signal solo
            # por el criterio normal de longitud/relleno, igual que cualquier otro
            # chunk -- estar duplicado no lo hace "poco informativo" por si mismo
            # (eso lo decide dedup.py mas adelante, marcando duplicate_of sobre la
            # fila resultante).
            flush()
            current, current_tokens = [], 0
            chunks.append(
                Chunk(
                    section_path=section_path,
                    chunk_index=-1,
                    n_chunks_in_doc=-1,
                    text=block.text,
                    n_tokens=n_tok,
                    content_type=block.content_type,
                    char_start=block.char_start,
                    char_end=block.char_end,
                    is_low_signal=(n_tok < _MIN_INFORMATIVE_TOKENS) or _looks_like_placeholder(block.text),
                )
            )
            continue

        if n_tok > config.max_tokens:
            # Este bloque puntual es gigante, pero hay otros bloques en la lista
            # (si estuviera solo, ya se resolvio arriba): se cierra lo que se venia
            # acumulando y este bloque se procesa aparte con una ventana, para no
            # mezclar los demas bloques con una ventana que no les corresponde.
            flush()
            current, current_tokens = [], 0
            for window_text, w_tok, _ in _window_fallback([block], config):
                chunks.append(
                    Chunk(
                        section_path=section_path, chunk_index=-1, n_chunks_in_doc=-1,
                        text=window_text, n_tokens=w_tok, content_type=block.content_type,
                        char_start=block.char_start, char_end=block.char_end, is_low_signal=False,
                    )
                )
            continue

        if current_tokens + n_tok > config.max_tokens and current:
            flush()
            current, current_tokens = [], 0

        current.append(block)
        current_tokens += n_tok

        if current_tokens >= config.target_tokens:
            flush()
            current, current_tokens = [], 0

    flush()
    return chunks


def chunk_sections(
    sections: list, config: ChunkingConfig, duplicate_block_hashes: frozenset[str] = frozenset()
) -> list[Chunk]:
    """Punto de entrada de este archivo: recibe TODAS las secciones de un
    documento (en orden) y devuelve la lista final de Chunk, con chunk_index y
    n_chunks_in_doc ya asignados (empezando en 0, consistente para todo el documento).
    """
    # Una seccion con menos tokens que este numero es candidata a juntarse con la
    # siguiente, en vez de emitirse sola (regla 4 de la explicacion al principio
    # del archivo).
    min_tokens_before_merge = max(1, config.target_tokens // 4)

    raw_chunks: list[Chunk] = []
    pending_blocks: list[Block] = []

    for idx, section in enumerate(sections):
        is_api_spec_section = bool(section.blocks) and all(b.content_type == "api_spec" for b in section.blocks)

        if is_api_spec_section:
            # Regla 5: nunca se junta con otra. Si habia bloques pendientes de una
            # seccion de Markdown chica anterior, se emiten solos aca (no tienen
            # con que juntarse, porque lo que sigue es un chunk de OpenAPI, de
            # naturaleza distinta).
            if pending_blocks:
                raw_chunks.extend(_pack_blocks(sections[idx - 1].section_path, pending_blocks, config, duplicate_block_hashes))
                pending_blocks = []
            raw_chunks.extend(_pack_blocks(section.section_path, section.blocks, config, duplicate_block_hashes))
            continue

        blocks_to_pack = pending_blocks + section.blocks
        section_tokens = sum(_count_tokens(b.text) for b in blocks_to_pack)
        is_last_section = idx == len(sections) - 1

        if section_tokens < min_tokens_before_merge and not is_last_section:
            # Seccion (que puede venir ya juntada con una anterior) que todavia es
            # chica, y que NO es la ultima del documento: se sigue cargando hacia
            # adelante, esperando a ver con que se junta.
            pending_blocks = blocks_to_pack
            continue

        raw_chunks.extend(_pack_blocks(section.section_path, blocks_to_pack, config, duplicate_block_hashes))
        pending_blocks = []

    # Si el documento entero termino con bloques pendientes sin juntar a nada (por
    # ejemplo, si la unica seccion del archivo es chica), se emiten solos igual --
    # ya se marcan is_low_signal dentro de _pack_blocks cuando corresponde, no se
    # descartan en silencio.
    if pending_blocks:
        raw_chunks.extend(_pack_blocks(sections[-1].section_path, pending_blocks, config, duplicate_block_hashes))

    n = len(raw_chunks)
    for i, chunk in enumerate(raw_chunks):
        chunk.chunk_index = i
        chunk.n_chunks_in_doc = n
    return raw_chunks
