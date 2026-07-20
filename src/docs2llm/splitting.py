"""Quinto paso del pipeline: decide que documentos van a train, cuales a val y
cuales a test (no decide esto chunk por chunk), respetando los grupos de
documentos conectados que calcula dedup.file_duplicate_components.

Por que se divide por documento y no por chunk suelto: dividir a nivel de chunk
podria dejar, por ejemplo, dos chunks casi identicos del MISMO archivo en splits
distintos -- una fuga de datos directa.

Por que se divide por GRUPO DE DOCUMENTOS CONECTADOS y no por archivo individual:
en catalog-portfolio-api se encontro una tabla de codigos de error repetida en 3
archivos -- eso obliga a que esos 3 archivos queden SIEMPRE en el mismo split. Si
no fuera asi, la misma tabla apareceria en train y en test a la vez.

Por que la cuota se reparte por CANTIDAD DE CHUNKS y no por cantidad de
documentos: los documentos de este corpus varian muchisimo en tamano (de 1 a mas
de 50 chunks). Repartir por cantidad de documentos ignora ese peso -- por ejemplo,
un proyecto con 5 documentos donde uno solo concentra 51 de los 75 chunks totales
puede terminar con el split real (medido en chunks, que es la unidad que de verdad
ve un modelo de retrieval/RAG o de fine-tuning) muy lejos de 60/20/20, aunque el
reparto "por cabeza" haya sido el correcto.

Por que el balanceo es GLOBAL (sobre todo el corpus a la vez) y no una cuota
60/20/20 recalculada de cero DENTRO de cada proyecto: la primera version de este
modulo hacia el balanceo por proyecto, pero eso tiene un problema real, que se
encontro corriendo el pipeline contra docs_raw completo: cada proyecto, evaluado
de forma aislada, manda "razonablemente" su documento mas pesado a la cuota mas
grande (train) -- pero con 10 proyectos haciendo eso al mismo tiempo, sin que
ninguno sepa cuanto ya absorbio train en los proyectos anteriores, el train
agregado terminaba en ~74% del corpus real (chunks), muy por encima del 60%
buscado. Por eso `assign_splits` pondera cada grupo de documentos por su
cantidad de chunks (`chunk_count_by_doc_id`) y corre un algoritmo greedy tipo
"longest-processing-time-first" (LPT, un heuristico clasico de balanceo de
carga) sobre TODOS los grupos del corpus a la vez, en una sola pasada: ordena
los grupos de mayor a menor peso y asigna cada uno, de a uno, al split que en
ESE momento (con el estado acumulado de TODO lo ya asignado, no solo de este
proyecto) este mas lejos de su cuota objetivo. Con el estado compartido entre
proyectos, si train ya absorbio varios documentos grandes de proyectos
anteriores, los proyectos que siguen dejan de mandarle todo lo grande a train
por default -- el balance final se mide sobre el corpus completo, no proyecto
por proyecto.

El reporte (mas abajo, en `report`) se arma DESPUES de decidir todo, y sigue
desglosado por proyecto -- sirve para auditar que cada proyecto tenga
representacion razonable en cada split, aunque la decision de asignacion en si
ya no reserva una cuota fija por proyecto.
"""

from __future__ import annotations

import random
from collections import defaultdict


def assign_splits(
    project_doc_ids: dict[str, list[str]],
    components: dict[str, str],
    seed: int,
    test_fraction: float,
    val_fraction: float,
    chunk_count_by_doc_id: dict[str, int],
) -> tuple[dict[str, str], dict[str, dict]]:
    """Asigna train/val/test sobre el corpus COMPLETO (aproximadamente 60/20/20
    EN CHUNKS, no en cantidad de documentos ni por proyecto), sin partir nunca un
    grupo de documentos conectados entre dos splits distintos. Devuelve
    (doc_id -> split, un reporte con el detalle por proyecto).

    Por que se escribio un repartidor propio en vez de usar herramientas ya hechas
    de scikit-learn (como GroupShuffleSplit o StratifiedGroupKFold): el
    comportamiento de esas herramientas frente a grupos de peso muy dispar (un
    documento de 1 chunk junto a uno de 51) es dificil de predecir y de explicar
    en vivo (ver notebooks/diagnostico.ipynb, seccion 7).

    `project_doc_ids`: {nombre_de_proyecto: [doc_id, ...]} -- todos los doc_id
    (archivos) de ese proyecto, en cualquier orden. Ya no se usa para calcular
    cuotas (el balanceo es global), solo para armar el reporte desglosado.
    `components`: doc_id -> id_del_grupo, viene de dedup.file_duplicate_components.
    `chunk_count_by_doc_id`: doc_id -> cantidad de chunks que aporta ese documento
    al corpus final -- el "peso" que usa el balanceo (ver el modulo, arriba).
    """
    # Peso (en chunks) de cada grupo de documentos conectados: la suma de los
    # chunks de todos los doc_id que caen en ese grupo, sin importar de que
    # proyecto vengan (un grupo casi siempre pertenece a un unico proyecto, pero
    # podria en principio abarcar mas de uno si dos proyectos distintos
    # compartieran contenido -- no se encontro ningun caso asi en docs_raw, pero
    # el calculo lo soporta igual).
    docs_by_component: dict[str, list[str]] = defaultdict(list)
    for doc_id, comp_id in components.items():
        docs_by_component[comp_id].append(doc_id)
    weight_of: dict[str, int] = {
        comp_id: sum(chunk_count_by_doc_id[d] for d in docs) for comp_id, docs in docs_by_component.items()
    }

    total_weight = sum(weight_of.values())
    targets = {
        "train": total_weight * (1 - test_fraction - val_fraction),
        "val": total_weight * val_fraction,
        "test": total_weight * test_fraction,
    }
    assigned_weight = {"train": 0.0, "val": 0.0, "test": 0.0}
    assigned: dict[str, str] = {}

    # LPT (longest-processing-time-first): se procesan los grupos de mayor a
    # menor peso, y cada uno se lo lleva el split que en ESE momento este mas
    # lejos de su cuota objetivo (deficit = objetivo - ya_asignado). El shuffle
    # (con semilla fija, para reproducibilidad) solo decide el desempate entre
    # grupos de peso identico -- el orden real de asignacion lo decide el peso.
    # El desempate entre splits con el MISMO deficit prefiere train, despues
    # test, despues val (ver el orden de la tupla en el max de mas abajo): en
    # un corpus con pocos grupos grandes, esto evita que val se termine
    # llevando de arrastre algo que "por las dudas" convendria mas en test.
    all_component_ids = sorted(weight_of)
    rng = random.Random(seed)
    shuffled = all_component_ids[:]
    rng.shuffle(shuffled)
    shuffled_pos = {c: i for i, c in enumerate(shuffled)}
    order = sorted(all_component_ids, key=lambda c: (-weight_of[c], shuffled_pos[c]))

    for c in order:
        split = max(("train", "test", "val"), key=lambda s: targets[s] - assigned_weight[s])
        assigned[c] = split
        assigned_weight[split] += weight_of[c]

    doc_id_to_split = {doc_id: assigned[comp_id] for doc_id, comp_id in components.items()}

    # El reporte se arma por proyecto DESPUES de decidir todo, a partir del
    # resultado ya calculado -- ya no hace falta procesar proyecto por proyecto
    # para decidir nada, pero desglosarlo asi sigue sirviendo para auditar si
    # alguno quedo sin representacion real en algun split.
    report: dict[str, dict] = {}
    for project in sorted(project_doc_ids):
        doc_ids = project_doc_ids[project]
        component_ids = sorted({components[d] for d in doc_ids})

        counts = {"train": 0, "val": 0, "test": 0}
        chunks = {"train": 0, "val": 0, "test": 0}
        for c in component_ids:
            counts[assigned[c]] += 1
            # Solo se suma la parte de chunks que pertenece a ESTE proyecto (no
            # el peso total del grupo, que podria incluir documentos de otro
            # proyecto en el caso raro de contenido compartido entre proyectos).
            chunks[assigned[c]] += sum(chunk_count_by_doc_id[d] for d in doc_ids if components[d] == c)

        warnings: list[str] = []
        if len(component_ids) == 1:
            warnings.append(
                f"'{project}' tiene un unico documento/grupo: su representacion en "
                f"val/test depende por completo del balance global del corpus, no "
                f"de una cuota propia."
            )
        for split in ("val", "test"):
            if counts[split] == 0:
                warnings.append(
                    f"'{project}' quedo sin representacion en {split} en este split -- "
                    f"tiene pocos documentos y/o son chicos frente al resto del corpus."
                )

        report[project] = {
            "n_documentos_unicos": len(component_ids),
            "n_documentos_totales": len(doc_ids),  # puede ser mayor a n_documentos_unicos si hay duplicados exactos entre archivos
            "conteo_por_split": counts,
            "chunks_por_split": chunks,
            "warnings": warnings,
        }

    return doc_id_to_split, report
