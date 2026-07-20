"""Tests de splitting.py -- el invariante mas critico del pipeline: ningun documento
(ni ningun contenido duplicado entre documentos) debe terminar repartido entre dos
splits distintos. Incluye un test de integracion contra el corpus REAL (no solo
unidades sinteticas) porque este invariante es exactamente lo que la prueba tecnica
pide defender ("un esquema y una separacion de datos que resistan ese uso").
"""

from __future__ import annotations

from docs2llm.config import load_canonical_sources, load_pipeline_config
from docs2llm.corpus_build import build_corpus_rows
from docs2llm.splitting import assign_splits


def test_proyecto_con_1_solo_documento_va_entero_a_train():
    doc_ids, report = assign_splits(
        project_doc_ids={"proyecto_solo": ["a.md"]},
        components={"a.md": "a.md"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={"a.md": 1},
    )
    print(f"\n[test] doc_ids: {doc_ids}")
    print(f"[test] warnings: {report['proyecto_solo']['warnings']}")
    assert doc_ids["a.md"] == "train"
    assert report["proyecto_solo"]["warnings"], "debe quedar loggeado que este proyecto no tiene representacion en val/test"


def test_proyecto_con_2_documentos_uno_a_train_otro_a_test():
    doc_ids, _ = assign_splits(
        project_doc_ids={"p": ["a.md", "b.md"]},
        components={"a.md": "a.md", "b.md": "b.md"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={"a.md": 1, "b.md": 1},
    )
    print(f"\n[test] doc_ids: {doc_ids}")
    assert sorted(doc_ids.values()) == ["test", "train"]


def test_proyecto_con_2_documentos_el_de_mas_chunks_va_a_train():
    """Con solo 2 documentos no hay margen para un 60/20/20 real, pero SI se puede
    elegir bien cual va a cada lado: el mas pesado (en chunks) a train -- la cuota
    mas grande -- y el mas liviano a test."""
    doc_ids, report = assign_splits(
        project_doc_ids={"p": ["chico.md", "grande.md"]},
        components={"chico.md": "chico.md", "grande.md": "grande.md"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={"chico.md": 2, "grande.md": 40},
    )
    print(f"\n[test] doc_ids: {doc_ids}")
    print(f"[test] chunks_por_split: {report['p']['chunks_por_split']}")
    assert doc_ids["grande.md"] == "train", "el documento con mas chunks debe ir a la cuota mas grande (train)"
    assert doc_ids["chico.md"] == "test"


def test_componente_compartido_nunca_se_reparte_entre_dos_documentos():
    """a.md y b.md comparten componente (contenido duplicado entre ellos) -- deben
    terminar SIEMPRE en el mismo split, sin importar cuantos otros documentos haya."""
    doc_ids, _ = assign_splits(
        project_doc_ids={"p": ["a.md", "b.md", "c.md", "d.md", "e.md"]},
        components={"a.md": "comp1", "b.md": "comp1", "c.md": "comp2", "d.md": "comp3", "e.md": "comp4"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={"a.md": 1, "b.md": 1, "c.md": 1, "d.md": 1, "e.md": 1},
    )
    print(f"\n[test] doc_ids: {doc_ids}")
    print(f"[test] split de a.md={doc_ids['a.md']!r}  split de b.md={doc_ids['b.md']!r}")
    assert doc_ids["a.md"] == doc_ids["b.md"], "a.md y b.md comparten componente, deben ir al mismo split"


def test_mismo_seed_da_siempre_el_mismo_resultado():
    kwargs = dict(
        project_doc_ids={"p": [f"doc{i}.md" for i in range(10)]},
        components={f"doc{i}.md": f"doc{i}.md" for i in range(10)},
        seed=7, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={f"doc{i}.md": (i % 5) + 1 for i in range(10)},
    )
    result_1, _ = assign_splits(**kwargs)
    result_2, _ = assign_splits(**kwargs)
    print(f"\n[test] corrida 1: {result_1}")
    print(f"[test] corrida 2: {result_2}")
    assert result_1 == result_2, "el split debe ser 100% reproducible con el mismo seed (requisito explicito del enunciado)"


def test_todos_los_proyectos_quedan_cubiertos_por_algun_split():
    doc_ids, report = assign_splits(
        project_doc_ids={"p": [f"doc{i}.md" for i in range(6)]},
        components={f"doc{i}.md": f"doc{i}.md" for i in range(6)},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={f"doc{i}.md": 1 for i in range(6)},
    )
    print(f"\n[test] conteo por split: {report['p']['conteo_por_split']}")
    assert set(doc_ids.values()) <= {"train", "val", "test"}
    assert len(doc_ids) == 6
    assert report["p"]["conteo_por_split"]["train"] >= 1, "con 6 documentos, train nunca deberia quedar vacio"


def test_balanceo_por_peso_favorece_documento_grande_en_train():
    """El caso real que motivo este cambio: un proyecto de 5 documentos donde uno
    solo concentra 51 de los 75 chunks totales (68% del peso). Repartir por
    CANTIDAD DE DOCUMENTOS ("por cabeza") no ve esta diferencia de tamano, y ese
    documento grande podria caer en cualquier split por puro sorteo -- de hecho asi
    paso en la corrida real contra docs_raw/ (fue a parar a test, ver README). El
    reparto por PESO (chunks) tiene que preferir la cuota mas grande (train) para
    el documento que domina el peso del proyecto."""
    doc_ids, report = assign_splits(
        project_doc_ids={"p": ["grande.md", "b.md", "c.md", "d.md", "e.md"]},
        components={"grande.md": "grande.md", "b.md": "b.md", "c.md": "c.md", "d.md": "d.md", "e.md": "e.md"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
        chunk_count_by_doc_id={"grande.md": 51, "b.md": 10, "c.md": 6, "d.md": 4, "e.md": 4},
    )
    print(f"\n[test] doc_ids: {doc_ids}")
    print(f"[test] conteo_por_split: {report['p']['conteo_por_split']}")
    print(f"[test] chunks_por_split: {report['p']['chunks_por_split']}")
    assert doc_ids["grande.md"] == "train", "el documento que concentra la mayoria del peso debe caer en la cuota mas grande (train)"
    assert report["p"]["conteo_por_split"]["val"] >= 1, "val no deberia quedar vacio, quedan 4 documentos chicos para repartir"
    assert report["p"]["conteo_por_split"]["test"] >= 1, "test no deberia quedar vacio, quedan 4 documentos chicos para repartir"


# --- Test de integracion contra el corpus REAL (no sintetico) -----------------------


def test_integracion_corpus_real_sin_fugas_entre_splits():
    """Corre el pipeline completo (build_corpus_rows) contra docs_raw/ real y verifica
    2 invariantes sobre el resultado final:
      1. Todo chunk de un mismo doc_id cae en el mismo split (invariante basico).
      2. Dos filas que comparten content_hash_sha256 (duplicado exacto detectado por
         dedup.py) SIEMPRE estan en el mismo split -- si esto fallara, la misma tabla
         de error-codes (u otro fragmento duplicado) podria aparecer en train Y en
         test simultaneamente: la fuga de datos textual que toda esta arquitectura
         existe para evitar.
    """
    print("\n[test] corriendo build_corpus_rows contra docs_raw/ real (esto imprime todo el pipeline)...")
    sources = load_canonical_sources()
    pipeline_config = load_pipeline_config()
    rows, _, _ = build_corpus_rows(sources, pipeline_config)
    print(f"[test] corpus real: {len(rows)} filas generadas")

    assert len(rows) > 0, "el corpus real no deberia estar vacio"

    # Invariante 1: un mismo doc_id nunca aparece con 2 splits distintos.
    split_by_doc = {}
    for row in rows:
        if row.doc_id in split_by_doc:
            assert split_by_doc[row.doc_id] == row.split, f"{row.doc_id} aparece en mas de un split"
        else:
            split_by_doc[row.doc_id] = row.split
    print(f"[test] invariante 1 OK: {len(split_by_doc)} documentos, ninguno repartido entre 2 splits")

    # Invariante 2: contenido duplicado (mismo hash) nunca queda repartido entre splits.
    split_by_hash: dict[str, str] = {}
    for row in rows:
        if row.content_hash_sha256 in split_by_hash:
            assert split_by_hash[row.content_hash_sha256] == row.split, (
                f"contenido duplicado (hash {row.content_hash_sha256[:12]}...) aparece en 2 splits distintos "
                f"-- fuga de datos real entre train/val/test"
            )
        else:
            split_by_hash[row.content_hash_sha256] = row.split
    print(f"[test] invariante 2 OK: {len(split_by_hash)} hashes distintos, ninguno repartido entre splits")

    # Sanity check adicional: los 3 splits deben existir y tener al menos 1 fila
    # (si target/test_fraction estuvieran mal configurados, val o test podrian quedar
    # vacios sin que ningun assert anterior lo detectara).
    splits_presentes = {row.split for row in rows}
    print(f"[test] splits presentes: {splits_presentes}")
    assert splits_presentes == {"train", "val", "test"}, f"se esperaban los 3 splits, se encontraron: {splits_presentes}"
