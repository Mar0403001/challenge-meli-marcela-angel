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

Por que se divide dentro de cada PROYECTO (con una cuota de aproximadamente
60/20/20 dentro de cada uno) y no con un 60/20/20 global sobre todo el corpus: con
solo 10 proyectos, un split global podria dejar un proyecto entero (por ejemplo
payment-promise-gateway, uno de los que mas contenido util tiene) completamente
afuera de train o de test. Dividiendo DENTRO de cada proyecto se garantiza que los
3 splits tengan representacion de (casi) todos los proyectos.
"""

from __future__ import annotations

import random


def assign_splits(
    project_doc_ids: dict[str, list[str]],
    components: dict[str, str],
    seed: int,
    test_fraction: float,
    val_fraction: float,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Asigna train/val/test por proyecto (aproximadamente 60/20/20), sin partir
    nunca un grupo de documentos conectados entre dos splits distintos. Devuelve
    (doc_id -> split, un reporte con el detalle por proyecto).

    Por que se escribio un repartidor propio en vez de usar herramientas ya hechas
    de scikit-learn (como GroupShuffleSplit o StratifiedGroupKFold): con varios
    proyectos que tienen un solo documento canonico, el comportamiento de esas
    herramientas en esos casos limite es dificil de predecir y de explicar en
    vivo (ver notebooks/diagnostico.ipynb, seccion 7).

    `project_doc_ids`: {nombre_de_proyecto: [doc_id, ...]} -- todos los doc_id
    (archivos) de ese proyecto, en cualquier orden (se ordenan aca mismo, para que
    el resultado sea siempre igual).
    `components`: doc_id -> id_del_grupo, viene de dedup.file_duplicate_components.
    """
    assigned: dict[str, str] = {}  # id_del_grupo -> split, se va completando proyecto por proyecto
    report: dict[str, dict] = {}

    for project in sorted(project_doc_ids):
        doc_ids = project_doc_ids[project]
        component_ids = sorted({components[d] for d in doc_ids})

        # Si un grupo de este proyecto ya quedo asignado al procesar OTRO proyecto
        # (esto solo pasaria si dos proyectos distintos comparten un chunk
        # identico -- no se encontro ningun caso asi en docs_raw, pero se
        # contempla de todas formas en vez de asumir que nunca puede pasar), se
        # respeta esa asignacion anterior y no se lo vuelve a contar en la cuota
        # de este proyecto.
        new_components = [c for c in component_ids if c not in assigned]

        # La semilla se arma combinando la semilla global con el nombre del
        # proyecto: esto hace que el resultado sea reproducible entre corridas
        # (misma semilla -> mismo resultado, siempre), pero evita que el orden
        # aleatorio sea el MISMO para cada proyecto (si no se hiciera esto, por
        # ejemplo "el primer archivo en orden alfabetico" podria terminar siempre
        # en test en todos los proyectos, solo por coincidencia del mezclado).
        rng = random.Random(f"{seed}:{project}")
        shuffled = new_components[:]
        rng.shuffle(shuffled)
        n = len(shuffled)

        warnings: list[str] = []
        if n == 0:
            pass  # todos los grupos de este proyecto ya estaban asignados (un caso raro)
        elif n == 1:
            assigned[shuffled[0]] = "train"
            warnings.append(
                f"Solo 1 documento/grupo unico en '{project}': va entero a train, "
                f"sin representacion en val/test para este proyecto."
            )
        elif n == 2:
            assigned[shuffled[0]] = "train"
            assigned[shuffled[1]] = "test"
            warnings.append(f"Solo 2 documentos/grupos en '{project}': 1 a train, 1 a test, ninguno a val.")
        else:
            n_test = max(1, round(n * test_fraction))
            n_val = max(1, round(n * val_fraction))
            # Garantiza que quede al menos 1 grupo en train, recortando primero de
            # val y despues de test si las fracciones configuradas dejarian train
            # vacio (esto puede pasar con un n chico, por ejemplo n=3 con
            # fracciones de 0.2 que al redondear dan 1 y 1... 3-1-1=1, esta bien;
            # pero se deja este resguardo generico para cualquier combinacion de
            # configuracion).
            while n_test + n_val >= n and (n_val > 1 or n_test > 1):
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            test_group = shuffled[:n_test]
            val_group = shuffled[n_test : n_test + n_val]
            train_group = shuffled[n_test + n_val :]
            for c in test_group:
                assigned[c] = "test"
            for c in val_group:
                assigned[c] = "val"
            for c in train_group:
                assigned[c] = "train"

        counts = {"train": 0, "val": 0, "test": 0}
        for c in component_ids:
            counts[assigned[c]] += 1
        report[project] = {
            "n_documentos_unicos": len(component_ids),
            "n_documentos_totales": len(doc_ids),  # puede ser mayor a n_documentos_unicos si hay duplicados exactos entre archivos
            "conteo_por_split": counts,
            "warnings": warnings,
        }

    doc_id_to_split = {doc_id: assigned[comp_id] for doc_id, comp_id in components.items()}
    return doc_id_to_split, report
