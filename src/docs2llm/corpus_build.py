"""Conecta todos los pasos: discovery -> normalize/parse -> chunk -> dedup -> split
-> corpus.jsonl.

Este archivo no tiene logica de negocio propia (esa vive en cada paso por
separado, para que cada uno se pueda entender y probar de forma aislada) -- solo
conecta las piezas en el orden correcto y arma las filas finales con todos los
campos de corpus.jsonl.
"""

from __future__ import annotations

import dataclasses
import re

from collections import Counter

from docs2llm import dedup, splitting
from docs2llm.config import PipelineConfig, ProjectSource
from docs2llm.discovery import SourceFile, discover_source_files
from docs2llm.normalize import normalize_markdown_text
from docs2llm.parse_markdown import parse_markdown_sections
from docs2llm.parse_openapi import parse_openapi_sections
from docs2llm.chunking import chunk_sections

# Heuristica de idioma sin depender de ninguna libreria externa: cuenta palabras
# funcionales (stopwords) tipicas de cada idioma. No es un detector de idioma
# robusto de proposito general (no distingue si una misma oracion mezcla los dos
# idiomas, ni reconoce idiomas fuera de es/en) -- alcanza para lo que
# efectivamente hay en docs_raw (archivos casi siempre en un solo idioma, con
# alguna excepcion puntual de datos de ejemplo en el otro idioma).
_ES_MARKERS_RE = re.compile(
    r"\b(el|la|los|las|de|del|para|con|una|uno|unos|unas|es|son|que|por|se|su|sus|"
    r"como|pero|más|esta|están|cada|todos|todas|debe|deben|así|también|sin|entre)\b",
    re.IGNORECASE,
)
_EN_MARKERS_RE = re.compile(
    r"\b(the|and|for|with|this|that|are|is|of|to|in|on|by|be|as|from|will|can|should|"
    r"must|when|not|have|has)\b",
    re.IGNORECASE,
)


def detect_language(text: str) -> str:
    """Heuristica de idioma (es/en/desconocido), contando cuantas palabras
    funcionales de cada idioma aparecen -- ver _ES_MARKERS_RE/_EN_MARKERS_RE arriba
    para el alcance real y las limitaciones que se aceptan (no detecta si una
    misma oracion mezcla idiomas, ni reconoce idiomas fuera de es/en).
    """
    es_hits = len(_ES_MARKERS_RE.findall(text))
    en_hits = len(_EN_MARKERS_RE.findall(text))
    if es_hits == en_hits:
        return "unknown"  # cubre el caso 0-0 (por ejemplo, un chunk que es solo una tabla de numeros) y los empates reales
    return "es" if es_hits > en_hits else "en"


@dataclasses.dataclass
class CorpusRow:
    doc_id: str  # estable por archivo fuente: es el relative_path (ver discovery.SourceFile)
    id: str  # doc_id + "#" + chunk_index, unico por fila
    project: str
    project_version: str
    source_path: str
    source_url_hint: str | None
    # La ruta de encabezados como texto legible ("H1 > H2 > H3"), no como lista
    # anidada: es mas facil de leer directamente en el archivo .jsonl (por ejemplo
    # al inspeccionarlo con la herramienta `jq`, o simplemente abriendo el
    # archivo) sin perder la informacion de jerarquia.
    section_path: str
    heading: str | None
    chunk_index: int
    n_chunks_in_doc: int
    char_start: int
    char_end: int
    text: str
    n_tokens: int
    language: str
    content_type: str
    is_low_signal: bool
    content_hash_sha256: str
    duplicate_of: str | None
    split: str


def _parse_file_into_sections(file: SourceFile) -> list:
    """Envia un SourceFile al lector correcto segun su file_kind: Markdown (que se
    limpia antes de leer) u OpenAPI (un lector dedicado, sin pasar por la limpieza
    de Markdown)."""
    raw_text = file.absolute_path.read_text(encoding="utf-8")
    if file.file_kind == "markdown":
        normalized_text = normalize_markdown_text(raw_text)
        return parse_markdown_sections(normalized_text, fallback_title=file.absolute_path.stem)
    else:  # "openapi_spec"
        return parse_openapi_sections(raw_text)


def build_corpus_rows(
    canonical_sources: dict[str, ProjectSource], pipeline_config: PipelineConfig
) -> tuple[list[CorpusRow], dict, list[tuple[str, str]]]:
    """Conecta todo el pipeline (leer -> armar chunks -> deduplicar -> dividir en
    splits) y devuelve las filas finales de corpus.jsonl, el reporte del split por
    proyecto, y los pares de chunks casi-duplicados encontrados (vacio si
    `pipeline_config.dedup.near_duplicate_check` esta apagado).

    Por que se hace en 2 pasadas (ver el bucle mas abajo): la primera pasada lee
    TODOS los archivos, para detectar que bloques se repiten en el corpus
    COMPLETO antes de armar los chunks de ninguno (ver
    notebooks/diagnostico.ipynb, seccion 6) -- un bloque solo se puede saber
    "duplicado" mirando el corpus entero, no un archivo a la vez.

    El reporte de deduplicacion no se devuelve aca: se arma despues, a partir de
    las propias filas (contando cuantas tienen duplicate_of distinto de nulo,
    agrupadas por content_hash_sha256) -- ver cli.py.
    """
    print("[build-corpus] descubriendo archivos en docs_raw/...")
    files = discover_source_files(canonical_sources)

    # Primera pasada: leer TODOS los archivos y separarlos en secciones/bloques
    # (todavia sin armar los chunks finales) para poder detectar que bloques
    # atomicos se repiten en MAS DE UN archivo del corpus completo -- eso no se
    # puede saber mirando un archivo a la vez. Un ejemplo real que motivo esto: el
    # boton HTML roto "Download X Template" se repite en 5 archivos de
    # catalog-portfolio-api, pero cada vez queda mezclado con texto distinto
    # alrededor; sin esta primera pasada, el chunk final de cada aparicion seria
    # distinto, y la deduplicacion por fila jamas los detectaria como duplicados
    # (esto se confirmo con este caso real, ver notebooks/diagnostico.ipynb, seccion 6).
    print(f"\n[build-corpus] primera pasada: parseando {len(files)} archivos y detectando bloques duplicados...")
    sections_by_file: list[tuple[SourceFile, list]] = []
    block_hash_counts: Counter[str] = Counter()
    for file in files:
        print(f"[build-corpus] leyendo {file.relative_path}")
        sections = _parse_file_into_sections(file)
        sections_by_file.append((file, sections))
        for section in sections:
            for block in section.blocks:
                block_hash_counts[dedup.compute_content_hash(block.text)] += 1
    duplicate_block_hashes = frozenset(h for h, count in block_hash_counts.items() if count > 1)
    print(f"[build-corpus] bloques con hash repetido en mas de un archivo: {len(duplicate_block_hashes)}")

    rows: list[CorpusRow] = []
    project_doc_ids: dict[str, list[str]] = {}

    print(f"\n[build-corpus] segunda pasada: armando chunks para {len(sections_by_file)} archivos...")
    for file, sections in sections_by_file:
        print(f"[build-corpus] chunkeando {file.relative_path}")
        chunks = chunk_sections(sections, pipeline_config.chunking, duplicate_block_hashes)

        if not chunks:
            # Caso real encontrado en docs_raw: zenith-keeper/guide/README.md es
            # literalmente un solo titulo sin ningun texto debajo (incluso despues
            # de que normalize.py arregle el "#API..." mal escrito, sigue sin
            # haber bloques que emitir -- ver parse_markdown.py, que descarta
            # secciones sin bloques). Un archivo asi no aporta NINGUNA fila al
            # corpus, asi que no debe contarse en project_doc_ids (si se contara,
            # quedaria sin entrada en `components`/`doc_id_to_split` mas abajo,
            # porque esos diccionarios se arman a partir de las filas reales --
            # eso era justo lo que causaba un error aca antes de agregar este chequeo).
            continue

        doc_id = file.relative_path
        project_doc_ids.setdefault(file.project, []).append(doc_id)

        for chunk in chunks:
            rows.append(
                CorpusRow(
                    doc_id=doc_id,
                    id=f"{doc_id}#{chunk.chunk_index}",
                    project=file.project,
                    project_version=file.project_version,
                    source_path=file.relative_path,
                    source_url_hint=file.source_url_hint,
                    section_path=" > ".join(chunk.section_path),
                    heading=chunk.section_path[-1] if chunk.section_path else None,
                    chunk_index=chunk.chunk_index,
                    n_chunks_in_doc=chunk.n_chunks_in_doc,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    n_tokens=chunk.n_tokens,
                    language=detect_language(chunk.text),
                    content_type=chunk.content_type,
                    is_low_signal=chunk.is_low_signal,
                    content_hash_sha256=dedup.compute_content_hash(chunk.text),
                    duplicate_of=None,  # se completa mas abajo, cuando ya se vieron TODAS las filas
                    split="",  # idem, se completa despues de calcular los grupos/splits
                )
            )

    # Deduplicacion: necesita ver el corpus COMPLETO a la vez (un chunk de un
    # archivo puede ser duplicado de uno de otro archivo distinto, ver dedup.py)
    # -- por eso corre despues del bucle de arriba, no adentro de el.
    print(f"\n[build-corpus] deduplicando {len(rows)} chunks...")
    dedup.assign_duplicate_of(rows)

    near_duplicate_pairs: list[tuple[str, str]] = []
    if pipeline_config.dedup.near_duplicate_check:
        print("[build-corpus] buscando duplicados aproximados (embeddings)...")
        near_duplicate_pairs = dedup.find_near_duplicate_pairs(rows, pipeline_config.dedup.near_duplicate_threshold)

    components = dedup.file_duplicate_components(rows, near_duplicate_pairs=near_duplicate_pairs)

    print(f"\n[build-corpus] asignando splits para {len(project_doc_ids)} proyectos...")
    doc_id_to_split, split_report = splitting.assign_splits(
        project_doc_ids=project_doc_ids,
        components=components,
        seed=pipeline_config.split.seed,
        test_fraction=pipeline_config.split.test_fraction,
        val_fraction=pipeline_config.split.val_fraction,
    )
    for row in rows:
        row.split = doc_id_to_split[row.doc_id]

    print(f"[build-corpus] listo: {len(rows)} filas finales\n")
    return rows, split_report, near_duplicate_pairs
