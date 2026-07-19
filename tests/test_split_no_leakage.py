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
    )
    assert doc_ids["a.md"] == "train"
    assert report["proyecto_solo"]["warnings"], "debe quedar loggeado que este proyecto no tiene representacion en val/test"


def test_proyecto_con_2_documentos_uno_a_train_otro_a_test():
    doc_ids, _ = assign_splits(
        project_doc_ids={"p": ["a.md", "b.md"]},
        components={"a.md": "a.md", "b.md": "b.md"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
    )
    assert sorted(doc_ids.values()) == ["test", "train"]


def test_componente_compartido_nunca_se_reparte_entre_dos_documentos():
    """a.md y b.md comparten componente (contenido duplicado entre ellos) -- deben
    terminar SIEMPRE en el mismo split, sin importar cuantos otros documentos haya."""
    doc_ids, _ = assign_splits(
        project_doc_ids={"p": ["a.md", "b.md", "c.md", "d.md", "e.md"]},
        components={"a.md": "comp1", "b.md": "comp1", "c.md": "comp2", "d.md": "comp3", "e.md": "comp4"},
        seed=42, test_fraction=0.2, val_fraction=0.2,
    )
    assert doc_ids["a.md"] == doc_ids["b.md"], "a.md y b.md comparten componente, deben ir al mismo split"


def test_mismo_seed_da_siempre_el_mismo_resultado():
    kwargs = dict(
        project_doc_ids={"p": [f"doc{i}.md" for i in range(10)]},
        components={f"doc{i}.md": f"doc{i}.md" for i in range(10)},
        seed=7, test_fraction=0.2, val_fraction=0.2,
    )
    result_1, _ = assign_splits(**kwargs)
    result_2, _ = assign_splits(**kwargs)
    assert result_1 == result_2, "el split debe ser 100% reproducible con el mismo seed (requisito explicito del enunciado)"


def test_todos_los_proyectos_quedan_cubiertos_por_algun_split():
    doc_ids, report = assign_splits(
        project_doc_ids={"p": [f"doc{i}.md" for i in range(6)]},
        components={f"doc{i}.md": f"doc{i}.md" for i in range(6)},
        seed=42, test_fraction=0.2, val_fraction=0.2,
    )
    assert set(doc_ids.values()) <= {"train", "val", "test"}
    assert len(doc_ids) == 6
    assert report["p"]["conteo_por_split"]["train"] >= 1, "con 6 documentos, train nunca deberia quedar vacio"


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
    sources = load_canonical_sources()
    pipeline_config = load_pipeline_config()
    rows, _ = build_corpus_rows(sources, pipeline_config)

    assert len(rows) > 0, "el corpus real no deberia estar vacio"

    # Invariante 1: un mismo doc_id nunca aparece con 2 splits distintos.
    split_by_doc = {}
    for row in rows:
        if row.doc_id in split_by_doc:
            assert split_by_doc[row.doc_id] == row.split, f"{row.doc_id} aparece en mas de un split"
        else:
            split_by_doc[row.doc_id] = row.split

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

    # Sanity check adicional: los 3 splits deben existir y tener al menos 1 fila
    # (si target/test_fraction estuvieran mal configurados, val o test podrian quedar
    # vacios sin que ningun assert anterior lo detectara).
    splits_presentes = {row.split for row in rows}
    assert splits_presentes == {"train", "val", "test"}, f"se esperaban los 3 splits, se encontraron: {splits_presentes}"
