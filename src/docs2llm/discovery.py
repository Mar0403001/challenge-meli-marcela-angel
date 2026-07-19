"""Primer paso del pipeline: recorre docs_raw/ y arma la lista de archivos que hay
que procesar.

Antes de leer una sola linea de contenido real, hay que decidir que archivos entran
al pipeline y cuales no -- separar la señal del ruido. Se aplican dos filtros, y los
dos estan explicados abajo (no son un "if" suelto sin contexto):

  1. Solo se mira la carpeta de la version elegida por proyecto (la que dice
     config/config.yaml, clave canonical_sources) -- asi se evita procesar (y de
     paso duplicar) el mismo contenido dos veces si un proyecto tiene varias
     versiones exportadas.
  2. Se excluyen los archivos de navegacion de docsify (_sidebar.md, _coverpage.md,
     etc.) y documentationTypes.json -- son archivos de configuracion de la
     herramienta que genera la documentacion, no contenido valioso sobre el
     dominio del proyecto.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from docs2llm.config import ProjectSource


@dataclasses.dataclass(frozen=True)
class SourceFile:
    """Un archivo fuente ya identificado y clasificado, listo para pasar a la
    etapa de parseo."""

    project: str
    project_version: str
    absolute_path: Path
    # Ruta relativa a la RAIZ DEL REPO (no a docs_raw/), por ejemplo
    # "docs_raw/catalog-portfolio-api/latest/guide/README.md". Este es el valor que
    # termina en el campo `source_path` de corpus.jsonl/qa.jsonl -- tiene que ser
    # relativa (no absoluta) para que la procedencia se pueda reproducir en
    # cualquier computadora.
    relative_path: str
    # "markdown" o "openapi_spec": decide que parser se usa mas adelante.
    # No hay otros tipos posibles porque docs_raw/ solo contiene archivos .md y
    # specs/swagger.yaml (ademas de documentationTypes.json, que ya se excluyo
    # antes de llegar aca).
    file_kind: str
    # La URL correspondiente dentro de documentationTypes.json (el campo "guide" o
    # "specifications"). Es una pista extra sobre de donde salio el archivo, ademas
    # de su ruta local: no apunta a un archivo puntual (el JSON trae una sola URL
    # por carpeta-version, no una por archivo), pero deja registrado de que
    # publicacion viene el contenido, mas alla de donde vive en el disco.
    source_url_hint: str | None


# Nombres de archivo que son configuracion de navegacion de docsify, no contenido
# real. _sidebar.md y _coverpage.md siguen la convencion propia de docsify (el
# prefijo "_" marca un archivo especial de la interfaz). "coverpage.md" (sin guion
# bajo) se agrega a mano porque el proyecto traffic-gate-api rompe esa convencion --
# se encontro que ese archivo cumple el mismo rol (una imagen de portada
# decorativa) aunque tenga un nombre distinto, asi que un filtro que solo mirara el
# prefijo "_" lo hubiera dejado pasar por error.
NAV_FILENAMES = {"_sidebar.md", "_coverpage.md", "_navbar.md", "coverpage.md"}

# Metadata pura de la herramienta de documentacion (URLs donde se publica), no
# contenido en si. Se lee solo para sacar source_url_hint (ver _load_url_hints),
# pero nunca se convierte en un fragmento del corpus.
METADATA_FILENAME = "documentationTypes.json"


def _load_url_hints(project_dir: Path) -> dict[str, str]:
    """Lee documentationTypes.json de una carpeta-version, si existe.

    Devuelve algo como {"guide": "https://...index.html", "specifications": "https://...yaml"}.
    Si el archivo no existe o esta corrupto, se devuelve un diccionario vacio en
    vez de fallar: esta metadata es un extra para rastrear el origen del contenido,
    no un requisito para que el pipeline funcione -- varios proyectos con
    contenido incompleto o placeholder igual tienen que poder procesarse.
    """
    metadata_path = project_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_source_files(canonical_sources: dict[str, ProjectSource]) -> list[SourceFile]:
    """Recorre cada carpeta de version ya elegida y devuelve la lista completa de
    archivos a procesar, sin la navegacion/metadata.

    Por que el orden en que se devuelven los archivos es siempre el mismo (primero
    por proyecto, despues por ruta): splitting.py necesita ese orden estable para
    que el split de train/val/test de siempre el mismo resultado con la misma
    semilla (ver notebooks/diagnostico.ipynb, seccion 7).
    """
    # Se calcula relativa a la raiz del repo (2 niveles arriba de docs_raw/), y no a
    # la carpeta desde la que se ejecuta el comando -- asi el campo relative_path no
    # cambia segun desde donde se invoque el CLI.
    repo_root = next(iter(canonical_sources.values())).path.parents[2] if canonical_sources else None

    files: list[SourceFile] = []
    for project, source in sorted(canonical_sources.items()):
        url_hints = _load_url_hints(source.path)
        files_before = len(files)

        # rglob("*") recorre TODAS las subcarpetas (guide/, specs/, guide/pages/,
        # etc.) sin necesidad de saber de antemano como esta organizado cada
        # proyecto -- se vio que la profundidad de carpetas varia mucho de un
        # proyecto a otro (por ejemplo catalog-portfolio-api tiene
        # guide/api-reference/, y otros proyectos no tienen subcarpetas en absoluto).
        for candidate in sorted(source.path.rglob("*")):
            if not candidate.is_file():
                continue
            if candidate.name in NAV_FILENAMES or candidate.name == METADATA_FILENAME:
                continue  # navegacion o metadata, no contenido real (ver arriba)

            if candidate.suffix == ".md":
                file_kind = "markdown"
                url_hint = url_hints.get("guide")
            elif candidate.suffix in (".yaml", ".yml"):
                # En los 3 proyectos que tienen este archivo, vive en
                # specs/swagger.yaml y es una especificacion OpenAPI real -- se
                # manda a un lector dedicado (parse_openapi.py), nunca al lector
                # generico de Markdown.
                file_kind = "openapi_spec"
                url_hint = url_hints.get("specifications")
            else:
                # docs_raw/ no deberia tener archivos con otra extension (se
                # confirmo al revisar el contenido), pero si en el futuro apareciera
                # uno nuevo, se lo ignora explicitamente en vez de intentar leerlo
                # como si fuera texto comun.
                continue

            files.append(
                SourceFile(
                    project=project,
                    project_version=source.canonical_version,
                    absolute_path=candidate,
                    relative_path=candidate.relative_to(repo_root).as_posix(),
                    file_kind=file_kind,
                    source_url_hint=url_hint,
                )
            )

        print(f"[discovery] {project} ({source.canonical_version}): {len(files) - files_before} archivos encontrados")

    print(f"[discovery] total: {len(files)} archivos en {len(canonical_sources)} proyectos")
    return files
