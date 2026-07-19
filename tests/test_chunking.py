"""Tests de chunking.py: construyen Section/Block a mano (sin archivos de por medio)
para poder forzar casos limite puntuales -- en particular el fallback de ventana para
un bloque oversized, que NO ocurre con los datos reales de docs_raw (se verifico al
correr build-corpus), asi que sin un test dedicado ese camino de codigo nunca se
ejercitaria en absoluto.
"""

from __future__ import annotations

from docs2llm.chunking import _count_tokens, chunk_sections
from docs2llm.config import ChunkingConfig
from docs2llm.parse_units import Block, Section

# Config chica y facil de razonar a mano para los tests (no la de config/config.yaml,
# que esta pensada para el corpus real, no para casos de test sinteticos).
_CFG = ChunkingConfig(target_tokens=40, max_tokens=80, overlap_tokens=10)


def _block(text: str, content_type: str = "prose", char_start: int = 0) -> Block:
    return Block(content_type=content_type, text=text, char_start=char_start, char_end=char_start + len(text))


def test_tabla_mas_grande_que_target_pero_bajo_max_no_se_parte():
    """Una tabla mas grande que target_tokens (el presupuesto BLANDO) pero todavia
    por debajo de max_tokens (el tope DURO) debe seguir siendo un solo chunk -- no
    hay que partirla solo porque ya paso el objetivo blando."""
    tabla = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | v{i} |" for i in range(6))
    assert _CFG.target_tokens < _count_tokens(tabla) < _CFG.max_tokens, "el tamano (en tokens reales) de la tabla de prueba debe quedar entre target y max para que el test tenga sentido"
    section = Section(heading="Tabla", level=1, section_path=("Tabla",), blocks=[_block(tabla, content_type="table")])

    chunks = chunk_sections([section], _CFG)

    assert len(chunks) == 1, "una tabla atomica bajo max_tokens no debe partirse solo por superar el target blando"
    assert chunks[0].text == tabla
    assert chunks[0].content_type == "table"


def test_tabla_oversized_permite_ventana_como_ultimo_recurso():
    """Si una tabla por si sola supera incluso max_tokens (caso extremo, no visto en
    docs_raw real), el pipeline prioriza no perder informacion sobre no partirla --
    cae al mismo fallback de ventana que cualquier otro bloque oversized. Es una
    excepcion documentada (regla 3 del modulo), no una violacion silenciosa."""
    tabla_gigante = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | valor_largo_{i} |" for i in range(40))
    section = Section(heading="Tabla", level=1, section_path=("Tabla",), blocks=[_block(tabla_gigante, content_type="table")])

    chunks = chunk_sections([section], _CFG)

    assert len(chunks) > 1, "una tabla que excede max_tokens por si sola debe usar el fallback de ventana"
    assert all(c.content_type == "table" for c in chunks)
    assert all(c.n_tokens <= _CFG.max_tokens for c in chunks)


def test_bloque_oversized_cae_a_ventana_con_overlap():
    """Un UNICO bloque que por si solo supera max_tokens (caso raro, no ocurre en
    docs_raw real) debe partirse en ventanas solapadas -- el unico camino del
    pipeline donde se corta texto a la mitad."""
    texto_gigante = "palabra " * 100  # ~100 tokens reales, por encima de max_tokens=80
    assert _count_tokens(texto_gigante) > _CFG.max_tokens, "el texto de prueba debe superar max_tokens para que el test tenga sentido"
    section = Section(heading="Gigante", level=1, section_path=("Gigante",), blocks=[_block(texto_gigante)])

    chunks = chunk_sections([section], _CFG)

    assert len(chunks) > 1, "un bloque oversized debe generar mas de un chunk via ventana"
    for c in chunks:
        assert c.n_tokens <= _CFG.max_tokens, "ninguna ventana debe exceder max_tokens"
    # Reconstruir el texto original a partir de las ventanas (sin el solapamiento)
    # deberia recuperar todo el contenido -- ninguna parte del bloque se pierde.
    assert chunks[0].text in texto_gigante
    assert chunks[-1].text in texto_gigante


def test_secciones_chicas_se_fusionan_con_la_siguiente():
    """Una seccion con muy poco contenido no debe emitirse como su propio chunk --
    se espera que se fusione con la seccion siguiente."""
    chica = Section(heading="Intro", level=1, section_path=("Intro",), blocks=[_block("Hola.")])
    grande = Section(
        heading="Detalle", level=1, section_path=("Detalle",),
        blocks=[_block("Contenido real y sustancioso sobre el tema, con varias frases de explicacion util.")],
    )

    chunks = chunk_sections([chica, grande], _CFG)

    assert len(chunks) == 1, "la seccion chica debia fusionarse con la siguiente, no emitirse sola"
    assert "Hola." in chunks[0].text
    assert "Contenido real" in chunks[0].text


def test_seccion_api_spec_nunca_se_fusiona_aunque_sea_chica():
    """Un chunk de OpenAPI (Section con blocks content_type='api_spec') debe quedar
    aislado siempre, incluso si es muy chico -- ver regla 5 del docstring del modulo."""
    schema_chico = Section(
        heading="Zones", level=1, section_path=("swagger.yaml", "schemas", "Zones"),
        blocks=[_block("Schema: Zones\nType: array", content_type="api_spec")],
    )
    endpoint = Section(
        heading="GET /x", level=1, section_path=("swagger.yaml", "paths", "GET /x"),
        blocks=[_block("Endpoint: GET /x\nSummary: algo", content_type="api_spec")],
    )

    chunks = chunk_sections([schema_chico, endpoint], _CFG)

    assert len(chunks) == 2, "las dos secciones api_spec deben quedar en chunks separados, nunca fusionadas"
    assert chunks[0].section_path == ("swagger.yaml", "schemas", "Zones")
    assert chunks[1].section_path == ("swagger.yaml", "paths", "GET /x")


def test_placeholder_se_marca_low_signal():
    section = Section(heading="TODO", level=1, section_path=("Seccion", "TODO"), blocks=[_block("TO DO.")])
    chunks = chunk_sections([section], _CFG)
    assert len(chunks) == 1
    assert chunks[0].is_low_signal is True


def test_bloque_duplicado_se_aisla_en_su_propio_chunk():
    """Si un bloque coincide con un hash de duplicate_block_hashes, debe salir aislado
    en su propio chunk aunque hubiera espacio de sobra para fusionarlo con vecinos."""
    from docs2llm.dedup import compute_content_hash

    fragmento_repetido = "Contenido que se repite identico en otro archivo del corpus."
    hash_repetido = compute_content_hash(fragmento_repetido)

    section = Section(
        heading="Mixta", level=1, section_path=("Mixta",),
        blocks=[_block("Texto previo distinto."), _block(fragmento_repetido), _block("Texto posterior distinto.")],
    )

    chunks_sin_dedup = chunk_sections([section], _CFG)
    chunks_con_dedup = chunk_sections([section], _CFG, duplicate_block_hashes=frozenset({hash_repetido}))

    # Sin el hash marcado, el bin-packing normal fusiona el fragmento con al menos
    # uno de sus vecinos (no queda solo) -- es el comportamiento que se quiere
    # contrastar, no importa en cuantos chunks termine partido el total.
    assert not any(c.text == fragmento_repetido for c in chunks_sin_dedup), (
        "sin marcar el hash, el fragmento no deberia quedar aislado por si solo (deberia venir fusionado con un vecino)"
    )

    # Con el hash marcado, el fragmento repetido debe salir en su propio chunk,
    # con su texto EXACTO, separado de los bloques vecinos.
    assert any(c.text == fragmento_repetido for c in chunks_con_dedup), "el fragmento duplicado debe aparecer aislado, texto exacto"


def test_n_chunks_in_doc_y_chunk_index_son_consistentes():
    s1 = Section(heading="A", level=1, section_path=("A",), blocks=[_block("Contenido de la primera seccion, con texto suficiente.")])
    s2 = Section(heading="B", level=1, section_path=("B",), blocks=[_block("Contenido de la segunda seccion, tambien con texto suficiente.")])

    chunks = chunk_sections([s1, s2], _CFG)

    assert len(chunks) >= 2
    assert all(c.n_chunks_in_doc == len(chunks) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
