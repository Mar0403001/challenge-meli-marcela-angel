"""Cuarto paso del pipeline: deduplicacion exacta de chunks, deteccion de
duplicados aproximados (near-duplicates) por similitud semantica, y deteccion de
que ARCHIVOS comparten contenido duplicado (para que splitting.py no los separe
entre distintos splits).

Este modulo trabaja sobre el corpus COMPLETO (todos los proyectos juntos), no
proyecto por proyecto: es mas simple de programar, y si dos proyectos distintos
llegaran a compartir un chunk identico (no se encontro ningun caso asi en
docs_raw, pero nada lo impide en principio), el resultado seria el mismo que si
se hubiera limitado el analisis a un solo proyecto -- osea, mirar todo el corpus a
la vez es una generalizacion segura, no una simplificacion que se pierda casos reales.

Este archivo responde tres preguntas distintas, con funciones separadas:
  1. "Este chunk puntual, ¿es un duplicado EXACTO de otro? ¿de cual?" -> assign_duplicate_of
  2. "¿Que pares de chunks son CASI iguales (mismo contenido con pequenos cambios,
     como una columna renombrada), sin llegar a ser identicos?" -> find_near_duplicate_pairs
  3. "¿Que ARCHIVOS terminan compartiendo contenido, exacto o casi-exacto, y por lo
     tanto tienen que quedar juntos en el mismo split de train/val/test?" ->
     file_duplicate_components

La pregunta 3 existe porque, en catalog-portfolio-api, se encontro una tabla de
codigos de error casi identica repetida en brands.md, brand-management/README.md
y troubleshooting/README.md -- si se dividiera el corpus por archivo sin tener
esto en cuenta, esa misma tabla terminaria en train Y en test al mismo tiempo: una
fuga de datos directa que un split ingenuo, por archivo, no puede detectar (ver
splitting.py).

Por que hace falta ademas del hash exacto: el hash es ciego a contenido que dice
"casi" lo mismo -- el caso real encontrado en 2p-revenue-optimizer-api es una
tabla con columnas renombradas (created_date -> date) entre versiones: mismo
contenido real, hash distinto porque el texto cambio. Un duplicado asi NO se
marca con `duplicate_of` (el texto no es literalmente identico, seria enganoso
llamarlo "duplicado exacto"), pero SI tiene que agrupar sus documentos en la
misma componente conexa para el split, por la misma razon que un duplicado
exacto: si quedara repartido entre train y test, el modelo veria en test una
version casi identica de algo que ya vio en train. Por eso la deteccion de
duplicados tiene dos capas desde el diseño: hash exacto para el caso barato y
determinista, embeddings semanticos (find_near_duplicate_pairs, mas abajo) para
el caso donde el contenido evoluciono pero sigue siendo, en esencia, lo mismo.
"""

from __future__ import annotations

import functools
import hashlib
import re
from collections import defaultdict
from typing import Protocol

# Junta corridas de espacios/saltos de linea en un unico espacio, y pasa todo a
# minuscula, PERO SOLO para calcular el hash -- el texto real que va a corpus.jsonl
# NO se toca (se guarda tal cual salio de chunking.py). Esto hace que dos chunks
# que solo difieren en el espaciado o en mayusculas/minusculas (por ejemplo, una
# tabla que se reformateo entre versiones) igual se detecten como duplicados
# EXACTOS, sin necesitar comparar el significado del texto (para eso esta
# find_near_duplicate_pairs, mas abajo).
_WHITESPACE_RE = re.compile(r"\s+")


def compute_content_hash(text: str) -> str:
    """Calcula un hash SHA-256 de `text`, despues de juntar espacios y pasar todo
    a minuscula -- esta es la clave que se usa para la deduplicacion EXACTA en
    todo el pipeline (tanto para chunks completos como para bloques atomicos).

    Por que normalizar espacios/mayusculas antes de calcular el hash: para
    detectar duplicados que solo difieren en formato (por ejemplo, una tabla
    realineada entre dos versiones del mismo documento), sin necesitar comparar
    el significado real del texto (para contenido que cambia MAS que el formato,
    ver find_near_duplicate_pairs).
    """
    normalized = _WHITESPACE_RE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _HasHashAndId(Protocol):
    id: str
    doc_id: str
    text: str
    content_type: str
    content_hash_sha256: str
    duplicate_of: str | None


def _duplicate_id_groups(rows: list[_HasHashAndId]) -> dict[str, list[str]]:
    """Arma un diccionario hash -> [ids], quedandose solo con los hashes que
    aparecen mas de una vez. El orden de la lista de ids sigue el orden de `rows`
    (que ya es siempre el mismo: discovery.py recorre docs_raw/ en orden, y
    chunking.py conserva el orden en que se lee cada documento) -- esto es lo que
    garantiza que "el primero que se ve es el que se toma como original" de
    siempre el mismo resultado entre corridas, en vez de depender del orden en que
    Python recorra un diccionario.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_hash[row.content_hash_sha256].append(row.id)
    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def assign_duplicate_of(rows: list[_HasHashAndId]) -> None:
    """Marca los duplicados EXACTOS de cada grupo de chunks repetidos, sin borrar
    ninguna fila.

    Por que marcar en vez de borrar: para que se pueda auditar en corpus.jsonl QUE
    se considero duplicado y de cual original, no solo que una fila desaparecio
    (ver notebooks/diagnostico.ipynb, seccion 6). Esta funcion modifica `rows`
    directamente (in-place): la primera fila de cada grupo queda con
    duplicate_of=None (es la version "original"); el resto apunta a ella.
    """
    rows_by_id = {row.id: row for row in rows}
    grupos = _duplicate_id_groups(rows)
    total_marcados = 0
    for ids in grupos.values():
        canonical_id = ids[0]
        for dup_id in ids[1:]:
            rows_by_id[dup_id].duplicate_of = canonical_id
            total_marcados += 1
    print(f"[dedup] {total_marcados} filas marcadas como duplicado exacto, en {len(grupos)} clusters")


# --- Duplicados aproximados (near-duplicates) por similitud de embeddings -----------
#
# Por que este modelo puntual (all-MiniLM-L6-v2) y no uno mas grande (ej.
# all-mpnet-base-v2, tambien de sentence-transformers): es el modelo de proposito
# general mas chico y rapido de esa libreria (384 dimensiones, ~90MB, entrenado de
# forma contrastiva sobre mas de mil millones de pares de oraciones -- ver su model
# card en Hugging Face, sentence-transformers/all-MiniLM-L6-v2), pensado
# justamente para similitud semantica/clustering de proposito general. Para ~334
# chunks no hace falta la precision extra de un modelo mas pesado: la tarea es
# separar "casi-duplicado" de "no relacionado" (una distincion gruesa, ver la
# calibracion en config/config.yaml), no un ranking fino de relevancia.
#
# El modelo (~90MB) esta vendorizado en vendor/sentence_transformers_cache/ (ver
# docs2llm/config.py, SENTENCE_TRANSFORMERS_MODEL_DIR) -- mismo patron que
# tiktoken: se descarga una sola vez durante el desarrollo y se guarda en el repo,
# para que build-corpus nunca dependa de la red en ninguna maquina.


@functools.lru_cache(maxsize=1)
def _embedding_model():
    """Carga el modelo de embeddings una sola vez por proceso (cachea el objeto,
    no solo el archivo en disco -- cargarlo de nuevo en cada llamada seria lento
    sin ninguna razon, el modelo no cambia durante una corrida)."""
    from sentence_transformers import SentenceTransformer

    from docs2llm.config import SENTENCE_TRANSFORMERS_MODEL_DIR

    return SentenceTransformer(str(SENTENCE_TRANSFORMERS_MODEL_DIR))


_CONTENT_TYPES_SIN_NEAR_DUPLICATE_CHECK = {"api_spec"}
# Los chunks de OpenAPI se generan con un formato propio muy rigido y repetido
# ("Endpoint: ...", "Operation ID: ...", "Parameters:", etc. -- ver
# parse_openapi.py) -- eso hace que dos operaciones GENUINAMENTE DISTINTAS (ej.
# "sign-in" vs "sign-up", o dos schemas con varios campos en comun) embedan con
# similitud alta solo por compartir la plantilla, no contenido real repetido. Se
# verifico contra el corpus real: a umbral 0.9, la mayoria de los pares
# encontrados en swagger.yaml eran endpoints/schemas distintos, no duplicados.
# El resto de content_type (prose/table/code) si mostro pares genuinos (ej. la
# misma tabla de config con "Password"/"Contraseña", o el mismo comando curl con
# distintos IDs de recurso), asi que la exclusion es especifica de api_spec, no
# general.


def find_near_duplicate_pairs(rows: list[_HasHashAndId], threshold: float) -> list[tuple[str, str]]:
    """Devuelve pares (id, id) de chunks cuyo contenido es semanticamente casi
    igual (similitud coseno de embeddings >= threshold), sin ser un duplicado
    EXACTO (eso ya lo cubre compute_content_hash/assign_duplicate_of).

    Por que embeddings y no una comparacion de texto mas simple (ej. diferencia
    caracter a caracter): el caso real que motivo esto (2p-revenue-optimizer-api,
    columnas renombradas entre versiones: created_date -> date) tiene texto
    literalmente distinto pero el MISMO significado -- una comparacion de texto
    puro no lo detectaria como relacionado, un embedding semantico si.

    Solo se comparan chunks del MISMO content_type (una tabla contra otra tabla,
    prosa contra prosa) -- comparar tipos distintos no aporta nada real y suma
    ruido. Los chunks "api_spec" se excluyen del todo, ver
    _CONTENT_TYPES_SIN_NEAR_DUPLICATE_CHECK arriba.

    Es O(n^2) comparaciones (se calcula la matriz de similitud completa) -- para
    los ~334 chunks de este corpus son ~55 mil pares, trivial. Para un corpus de
    millones de chunks esto dejaria de ser practico y haria falta un indice
    aproximado (ej. FAISS), pero no se justifica esa complejidad para este tamano.
    """
    candidatos = [row for row in rows if row.content_type not in _CONTENT_TYPES_SIN_NEAR_DUPLICATE_CHECK]
    if len(candidatos) < 2:
        return []

    from sklearn.metrics.pairwise import cosine_similarity

    model = _embedding_model()
    textos = [row.text for row in candidatos]
    embeddings = model.encode(textos, show_progress_bar=False, convert_to_numpy=True)
    matriz_similitud = cosine_similarity(embeddings)

    pares: list[tuple[str, str]] = []
    n = len(candidatos)
    for i in range(n):
        for j in range(i + 1, n):
            if candidatos[i].content_type != candidatos[j].content_type:
                continue
            if candidatos[i].content_hash_sha256 == candidatos[j].content_hash_sha256:
                continue  # ya es un duplicado EXACTO, ya esta cubierto por assign_duplicate_of
            if matriz_similitud[i][j] >= threshold:
                pares.append((candidatos[i].id, candidatos[j].id))

    print(f"[dedup] {len(pares)} pares casi-duplicados encontrados (umbral={threshold}, modelo all-MiniLM-L6-v2)")
    return pares


class _UnionFind:
    """Una estructura "union-find" (o "conjuntos disjuntos"): sirve para agrupar
    elementos que estan conectados entre si, aunque sea de forma indirecta (por
    ejemplo, A esta conectado con B, y B con C, entonces A/B/C quedan en el mismo
    grupo aunque A y C nunca se hayan comparado directamente). Es una version
    minima, sin las optimizaciones mas avanzadas (como "union por rango"), porque
    la cantidad de archivos de este corpus es chica -- no hace falta optimizar
    mas alla de que el codigo se entienda bien, para un desafio que se resuelve
    en un tiempo acotado.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Siempre se une hacia el representante que es menor en orden alfabetico:
        # asi el resultado final es siempre el mismo, sin importar en que orden se
        # procesen los grupos de duplicados (esto es necesario para que el split,
        # que depende de este resultado, sea reproducible entre corridas).
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


def file_duplicate_components(
    rows: list[_HasHashAndId],
    near_duplicate_pairs: list[tuple[str, str]] = (),
) -> dict[str, str]:
    """Devuelve un diccionario doc_id -> id_del_grupo (el doc_id que representa a
    todo el grupo).

    Dos documentos quedan en el mismo grupo si comparten AL MENOS un chunk con el
    mismo hash exacto, O si `near_duplicate_pairs` (opcional, ver
    find_near_duplicate_pairs) marca alguno de sus chunks como casi-duplicado de
    un chunk del otro documento. `splitting.py` usa esto como la unidad real para
    el split: todos los doc_id de un mismo grupo van al mismo split, sin excepcion.

    Por que los near-duplicates tambien unen documentos, con el mismo criterio que
    los duplicados exactos: si dos documentos comparten contenido CASI identico
    (ej. la misma tabla con una columna renombrada), dejarlos en splits distintos
    es casi tan riesgoso como un duplicado exacto -- el modelo veria en test una
    version levemente distinta de algo que ya vio en train.
    """
    uf = _UnionFind()
    doc_id_by_row_id = {row.id: row.doc_id for row in rows}

    for ids in _duplicate_id_groups(rows).values():
        doc_ids = sorted({doc_id_by_row_id[i] for i in ids})
        for other in doc_ids[1:]:
            uf.union(doc_ids[0], other)

    for id_a, id_b in near_duplicate_pairs:
        uf.union(doc_id_by_row_id[id_a], doc_id_by_row_id[id_b])

    # Todo doc_id existe como su propio grupo, aunque no comparta ningun duplicado
    # con nadie (el caso mas comun: la mayoria de los archivos no repiten contenido).
    all_doc_ids = {row.doc_id for row in rows}
    resultado = {doc_id: uf.find(doc_id) for doc_id in all_doc_ids}
    n_componentes = len(set(resultado.values()))
    print(f"[dedup] {len(all_doc_ids)} documentos agrupados en {n_componentes} componentes conexas")
    return resultado
