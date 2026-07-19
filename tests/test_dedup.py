"""Tests de dedup.py: hash exacto (con normalizacion de espacios/mayusculas) y
componentes conexas entre archivos que comparten contenido duplicado.
"""

from __future__ import annotations

import dataclasses

from docs2llm.dedup import assign_duplicate_of, compute_content_hash, file_duplicate_components


@dataclasses.dataclass
class _Row:
    """Doble minimo de CorpusRow -- solo los campos que dedup.py realmente usa
    (ver el Protocol _HasHashAndId en dedup.py), para no depender de corpus_build.py
    en estos tests unitarios."""

    id: str
    doc_id: str
    content_hash_sha256: str
    duplicate_of: str | None = None


def test_hash_normaliza_espacios_y_mayusculas():
    """Dos textos que difieren solo en espaciado/capitalizacion deben producir el
    MISMO hash -- esto es lo que permite detectar duplicados aunque el reformateo
    de una tabla (ver vendor-stockkeeper-api, columnas realineadas entre versiones)
    cambie el espaciado sin cambiar el contenido real."""
    a = "Hola   Mundo\n\ncon    espacios raros"
    b = "hola mundo con espacios raros"
    assert compute_content_hash(a) == compute_content_hash(b)


def test_hash_distingue_contenido_realmente_distinto():
    assert compute_content_hash("contenido A") != compute_content_hash("contenido B")


def test_assign_duplicate_of_marca_todas_menos_la_primera():
    hash_x = compute_content_hash("contenido repetido")
    hash_y = compute_content_hash("contenido unico")
    rows = [
        _Row(id="doc1#0", doc_id="doc1", content_hash_sha256=hash_x),
        _Row(id="doc2#0", doc_id="doc2", content_hash_sha256=hash_x),
        _Row(id="doc3#0", doc_id="doc3", content_hash_sha256=hash_x),
        _Row(id="doc4#0", doc_id="doc4", content_hash_sha256=hash_y),  # no duplicado, unico en su hash
    ]

    assign_duplicate_of(rows)

    assert rows[0].duplicate_of is None, "la primera fila vista de un cluster de duplicados es la canonica"
    assert rows[1].duplicate_of == "doc1#0"
    assert rows[2].duplicate_of == "doc1#0"
    assert rows[3].duplicate_of is None, "una fila sin ningun duplicado no debe marcarse"


def test_file_duplicate_components_une_archivos_que_comparten_hash():
    """3 archivos comparten un chunk con el mismo hash (ej. la tabla de error-codes
    de catalog-portfolio-api repetida en brands.md/brand-management/troubleshooting)
    -> deben terminar en el MISMO componente. Un 4to archivo sin nada en comun con
    los demas debe quedar en su propio componente, aislado."""
    hash_compartido = compute_content_hash("tabla de error codes compartida")
    hash_unico_a = compute_content_hash("contenido exclusivo de archivo A")
    hash_unico_d = compute_content_hash("contenido exclusivo de archivo D, sin relacion con nada")

    rows = [
        _Row(id="a#0", doc_id="archivo_a.md", content_hash_sha256=hash_unico_a),
        _Row(id="a#1", doc_id="archivo_a.md", content_hash_sha256=hash_compartido),
        _Row(id="b#0", doc_id="archivo_b.md", content_hash_sha256=hash_compartido),
        _Row(id="c#0", doc_id="archivo_c.md", content_hash_sha256=hash_compartido),
        _Row(id="d#0", doc_id="archivo_d.md", content_hash_sha256=hash_unico_d),
    ]

    components = file_duplicate_components(rows)

    assert components["archivo_a.md"] == components["archivo_b.md"] == components["archivo_c.md"], (
        "los 3 archivos que comparten el chunk duplicado deben quedar en el mismo componente"
    )
    assert components["archivo_d.md"] != components["archivo_a.md"], "un archivo sin contenido compartido no debe unirse a otros componentes"


def test_file_duplicate_components_determinista_entre_corridas():
    """Mismo input -> mismo resultado, sin importar el orden interno de iteracion de
    dicts/sets (requisito de reproducibilidad: splitting.py depende de esto)."""
    hash_compartido = compute_content_hash("contenido compartido")
    rows = [
        _Row(id="z#0", doc_id="z.md", content_hash_sha256=hash_compartido),
        _Row(id="a#0", doc_id="a.md", content_hash_sha256=hash_compartido),
    ]

    result_1 = file_duplicate_components(rows)
    result_2 = file_duplicate_components(list(reversed(rows)))

    assert result_1 == result_2, "el resultado no deberia depender del orden de entrada de las filas"
