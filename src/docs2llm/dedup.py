"""Cuarto paso del pipeline: deduplicacion exacta de chunks, y deteccion de que
ARCHIVOS comparten contenido duplicado (para que splitting.py no los separe entre
distintos splits).

Este modulo trabaja sobre el corpus COMPLETO (todos los proyectos juntos), no
proyecto por proyecto: es mas simple de programar, y si dos proyectos distintos
llegaran a compartir un chunk identico (no se encontro ningun caso asi en
docs_raw, pero nada lo impide en principio), el resultado seria el mismo que si
se hubiera limitado el analisis a un solo proyecto -- osea, mirar todo el corpus a
la vez es una generalizacion segura, no una simplificacion que se pierda casos reales.

Este archivo responde dos preguntas distintas, con dos funciones separadas:
  1. "Este chunk puntual, ¿es un duplicado exacto de otro? ¿de cual?" -> assign_duplicate_of
  2. "¿Que ARCHIVOS terminan compartiendo contenido, aunque sea parcialmente, y por
     lo tanto tienen que quedar juntos en el mismo split de train/val/test?" ->
     file_duplicate_components

La pregunta 2 existe porque, en catalog-portfolio-api, se encontro una tabla de
codigos de error casi identica repetida en brands.md, brand-management/README.md
y troubleshooting/README.md -- si se dividiera el corpus por archivo sin tener
esto en cuenta, esa misma tabla terminaria en train Y en test al mismo tiempo: una
fuga de datos directa que un split ingenuo, por archivo, no puede detectar (ver
splitting.py).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Protocol

# Junta corridas de espacios/saltos de linea en un unico espacio, y pasa todo a
# minuscula, PERO SOLO para calcular el hash -- el texto real que va a corpus.jsonl
# NO se toca (se guarda tal cual salio de chunking.py). Esto hace que dos chunks
# que solo difieren en el espaciado o en mayusculas/minusculas (por ejemplo, una
# tabla que se reformateo entre versiones) igual se detecten como duplicados, sin
# necesitar comparar el significado del texto (eso queda para el extra opcional de
# duplicados aproximados por embeddings, que no se implemento en este MVP).
_WHITESPACE_RE = re.compile(r"\s+")


def compute_content_hash(text: str) -> str:
    """Calcula un hash SHA-256 de `text`, despues de juntar espacios y pasar todo
    a minuscula -- esta es la clave que se usa para la deduplicacion exacta en
    todo el pipeline (tanto para chunks completos como para bloques atomicos).

    Por que normalizar espacios/mayusculas antes de calcular el hash: para
    detectar duplicados que solo difieren en formato (por ejemplo, una tabla
    realineada entre dos versiones del mismo documento), sin necesitar comparar
    el significado real del texto (eso queda para el extra opcional de
    duplicados aproximados por embeddings, no implementado en este MVP).
    """
    normalized = _WHITESPACE_RE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class _HasHashAndId(Protocol):
    id: str
    doc_id: str
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
    """Marca los duplicados exactos de cada grupo de chunks repetidos, sin borrar
    ninguna fila.

    Por que marcar en vez de borrar: para que se pueda auditar en corpus.jsonl QUE
    se considero duplicado y de cual original, no solo que una fila desaparecio
    (ver notebooks/diagnostico.ipynb, seccion 6). Esta funcion modifica `rows`
    directamente (in-place): la primera fila de cada grupo queda con
    duplicate_of=None (es la version "original"); el resto apunta a ella.
    """
    rows_by_id = {row.id: row for row in rows}
    for ids in _duplicate_id_groups(rows).values():
        canonical_id = ids[0]
        for dup_id in ids[1:]:
            rows_by_id[dup_id].duplicate_of = canonical_id


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


def file_duplicate_components(rows: list[_HasHashAndId]) -> dict[str, str]:
    """Devuelve un diccionario doc_id -> id_del_grupo (el doc_id que representa a
    todo el grupo).

    Dos documentos quedan en el mismo grupo si comparten AL MENOS un chunk con el
    mismo hash exacto. `splitting.py` usa esto como la unidad real para el split:
    todos los doc_id de un mismo grupo van al mismo split, sin excepcion.
    """
    uf = _UnionFind()
    doc_id_by_row_id = {row.id: row.doc_id for row in rows}

    for ids in _duplicate_id_groups(rows).values():
        doc_ids = sorted({doc_id_by_row_id[i] for i in ids})
        for other in doc_ids[1:]:
            uf.union(doc_ids[0], other)

    # Todo doc_id existe como su propio grupo, aunque no comparta ningun duplicado
    # con nadie (el caso mas comun: la mayoria de los archivos no repiten contenido).
    all_doc_ids = {row.doc_id for row in rows}
    return {doc_id: uf.find(doc_id) for doc_id in all_doc_ids}
