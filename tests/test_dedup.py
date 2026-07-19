"""Tests de dedup.py: hash exacto (con normalizacion de espacios/mayusculas),
duplicados aproximados por embeddings, y componentes conexas entre archivos que
comparten contenido duplicado (exacto o aproximado).
"""

from __future__ import annotations

import dataclasses

from docs2llm.dedup import (
    assign_duplicate_of,
    compute_content_hash,
    file_duplicate_components,
    find_near_duplicate_pairs,
)


@dataclasses.dataclass
class _Row:
    """Doble minimo de CorpusRow -- solo los campos que dedup.py realmente usa
    (ver el Protocol _HasHashAndId en dedup.py), para no depender de corpus_build.py
    en estos tests unitarios."""

    id: str
    doc_id: str
    content_hash_sha256: str
    duplicate_of: str | None = None
    text: str = ""
    content_type: str = "prose"


def test_hash_normaliza_espacios_y_mayusculas():
    """Dos textos que difieren solo en espaciado/capitalizacion deben producir el
    MISMO hash -- esto es lo que permite detectar duplicados aunque el reformateo
    de una tabla (ver vendor-stockkeeper-api, columnas realineadas entre versiones)
    cambie el espaciado sin cambiar el contenido real."""
    a = "Hola   Mundo\n\ncon    espacios raros"
    b = "hola mundo con espacios raros"
    hash_a, hash_b = compute_content_hash(a), compute_content_hash(b)
    print(f"\n[test] hash(a)={hash_a[:12]}...  hash(b)={hash_b[:12]}...")
    assert hash_a == hash_b


def test_hash_distingue_contenido_realmente_distinto():
    hash_a = compute_content_hash("contenido A")
    hash_b = compute_content_hash("contenido B")
    print(f"\n[test] hash(A)={hash_a[:12]}...  hash(B)={hash_b[:12]}...")
    assert hash_a != hash_b


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
    print(f"\n[test] duplicate_of por fila: {[(r.id, r.duplicate_of) for r in rows]}")

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
    print(f"\n[test] componentes: {components}")

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
    print(f"\n[test] orden normal: {result_1}")
    print(f"[test] orden invertido: {result_2}")

    assert result_1 == result_2, "el resultado no deberia depender del orden de entrada de las filas"


# --- Duplicados aproximados (near-duplicates) por embeddings ------------------------
# Estos tests SI cargan el modelo real (vendorizado en
# vendor/sentence_transformers_cache/) -- es chico y no necesita red, asi que correr
# el modelo de verdad (en vez de un stub) es mas representativo que simularlo, con
# un costo de tiempo bajo (unos pocos segundos).


def test_columnas_renombradas_se_detectan_como_near_duplicate():
    """El caso real que motivo esta funcion: 2p-revenue-optimizer-api tiene una
    tabla con columnas renombradas entre versiones (created_date -> date) -- mismo
    significado, hash distinto. La similitud semantica tiene que detectarlo."""
    tabla_v1 = "| created_date | productive_date | status |\n|---|---|---|\n| 2024-01-01 | 2024-01-05 | ok |"
    tabla_v2 = "| date | version | status |\n|---|---|---|\n| 2024-01-01 | 2024-01-05 | ok |"
    texto_no_relacionado = "El servidor web configurado en este proyecto es nginx, con balanceo de carga round-robin."

    rows = [
        _Row(id="a#0", doc_id="a.md", content_hash_sha256=compute_content_hash(tabla_v1), text=tabla_v1, content_type="table"),
        _Row(id="b#0", doc_id="b.md", content_hash_sha256=compute_content_hash(tabla_v2), text=tabla_v2, content_type="table"),
        _Row(id="c#0", doc_id="c.md", content_hash_sha256=compute_content_hash(texto_no_relacionado), text=texto_no_relacionado, content_type="prose"),
    ]

    pares = find_near_duplicate_pairs(rows, threshold=0.6)
    print(f"\n[test] pares casi-duplicados encontrados: {pares}")

    assert ("a#0", "b#0") in pares, "las dos versiones de la tabla (con columnas renombradas) deben detectarse como casi-duplicadas"
    assert not any("c#0" in par for par in pares), "el texto sin relacion no deberia aparecer en ningun par"


def test_duplicado_exacto_no_se_repite_como_near_duplicate():
    """Un par que ya es duplicado EXACTO (mismo hash) no debe aparecer tambien en
    find_near_duplicate_pairs -- ya esta cubierto por assign_duplicate_of, listarlo
    de nuevo aca seria redundante."""
    texto = "Contenido identico, palabra por palabra, en dos archivos distintos."
    rows = [
        _Row(id="a#0", doc_id="a.md", content_hash_sha256=compute_content_hash(texto), text=texto),
        _Row(id="b#0", doc_id="b.md", content_hash_sha256=compute_content_hash(texto), text=texto),
    ]

    pares = find_near_duplicate_pairs(rows, threshold=0.9)
    print(f"\n[test] pares (deberia ser vacio, ya es duplicado exacto): {pares}")

    assert pares == []


def test_near_duplicate_pairs_tambien_unen_componentes_para_el_split():
    """file_duplicate_components debe unir documentos por near_duplicate_pairs
    ademas de por hash exacto -- es la conexion real con splitting.py."""
    rows = [
        _Row(id="a#0", doc_id="a.md", content_hash_sha256=compute_content_hash("texto A")),
        _Row(id="b#0", doc_id="b.md", content_hash_sha256=compute_content_hash("texto B")),
        _Row(id="c#0", doc_id="c.md", content_hash_sha256=compute_content_hash("texto C, sin relacion")),
    ]

    components = file_duplicate_components(rows, near_duplicate_pairs=[("a#0", "b#0")])
    print(f"\n[test] componentes con near-duplicate pair (a,b): {components}")

    assert components["a.md"] == components["b.md"], "a.md y b.md deben unirse por el par casi-duplicado"
    assert components["c.md"] != components["a.md"], "c.md no deberia unirse a nada, no comparte ningun par"
