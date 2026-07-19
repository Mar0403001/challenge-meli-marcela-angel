"""Lee y valida config/config.yaml: tiene una clave "canonical_sources" (qué versión
usar por proyecto) y una clave "pipeline" (los números con los que corre el
pipeline: tamaño de chunk, fracciones del split, etc.). Antes eran dos archivos
separados; se juntaron en uno porque las dos cosas son "configuración que vive
fuera del código", sin otra razón real para estar separadas.

El resto del pipeline siempre importa las funciones de este archivo en vez de leer
el YAML directamente, por dos motivos: si alguien escribe mal una clave, el error
aparece acá mismo con un mensaje claro (ver load_pipeline_config), y ningún otro
módulo necesita escribir una ruta de archivo a mano.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

# PyYAML (el paquete se instala como "PyYAML" pero se importa como "yaml") es la
# libreria que lee archivos .yaml y los convierte en diccionarios/listas de Python.
# Se usa tanto para config/config.yaml como, en otro modulo, para los swagger.yaml.
import yaml

# La raiz del repo se calcula a partir de la ubicacion de ESTE archivo, en vez de
# escribirla como texto fijo. src/docs2llm/config.py -> subiendo 2 carpetas
# (docs2llm/, src/) se llega a la raiz. Gracias a esto el pipeline funciona igual
# sin importar en que computadora o carpeta se haya clonado el repo (un requisito
# explicito del desafio: "sin rutas locales escritas a mano").
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_RAW_DIR = REPO_ROOT / "docs_raw"
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

# El archivo de codificacion de tiktoken (cl100k_base) esta guardado dentro del
# propio repositorio -- ver chunking.py para la explicacion completa de por que:
# en resumen, sin esto, tiktoken intentaria descargar ese archivo de internet la
# primera vez que se usa, y esa descarga puede fallar por restricciones de red o
# de certificados segun la maquina (se comprobo que efectivamente fallaba). Con el
# archivo ya guardado en TIKTOKEN_CACHE_DIR, tiktoken lo lee directo del disco y
# nunca necesita internet. Esto se define aca, al importar el modulo (no dentro de
# una funcion), porque config.py siempre se importa antes que chunking.py, sin
# importar desde donde se arranque el pipeline (la terminal, los tests, el notebook).
TIKTOKEN_CACHE_DIR = REPO_ROOT / "vendor" / "tiktoken_cache"
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(TIKTOKEN_CACHE_DIR))


def load_config_yaml(path: Path | None = None) -> dict:
    """Lee config/config.yaml tal cual, sin procesar (un diccionario con las claves
    `canonical_sources` y `pipeline`). La usan load_canonical_sources y
    load_pipeline_config de aca abajo, y tambien cli.py directamente cuando necesita
    el archivo completo (ver el subcomando validate-canonical-sources).
    """
    path = path or CONFIG_PATH
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclasses.dataclass(frozen=True)
class ProjectSource:
    """La version elegida para un proyecto, ya revisada y lista para usar."""

    project: str
    # El nombre exacto de la carpeta de version elegida, ej. "latest", "0.2.2",
    # "0.0.10-stock-consumer". Se guarda tal cual viene, sin intentar interpretarlo
    # como un numero de version real, porque varias carpetas no siguen ese formato
    # (ej. "0.0.1-test-bench-v2a").
    canonical_version: str
    reason: str
    # La ruta ya armada y confirmada que existe: docs_raw/<project>/<canonical_version>.
    path: Path


def load_canonical_sources(path: Path | None = None) -> dict[str, ProjectSource]:
    """Lee la clave `canonical_sources` de config.yaml y devuelve un diccionario
    {nombre_de_proyecto: ProjectSource}.

    Ademas confirma que la carpeta mencionada exista de verdad en docs_raw/. Si
    alguien escribe mal el nombre de una version en el YAML (por ejemplo
    "0.0.10-stok-consumer" con una letra de menos), preferimos que el error aparezca
    aca mismo, con un mensaje claro, en vez de mas adelante como un error generico de
    "archivo no encontrado" en medio de leer otro proyecto distinto.
    """
    raw = load_config_yaml(path)["canonical_sources"]

    sources: dict[str, ProjectSource] = {}
    for project, entry in raw["projects"].items():
        canonical_version = str(entry["canonical"])
        project_path = DOCS_RAW_DIR / project / canonical_version
        if not project_path.is_dir():
            raise FileNotFoundError(
                f"config.yaml (canonical_sources) dice que el proyecto '{project}' usa la "
                f"version '{canonical_version}', pero esa carpeta no existe en docs_raw/ "
                f"(se busco en: {project_path}). Revisa que el nombre de la carpeta este bien escrito."
            )
        sources[project] = ProjectSource(
            project=project,
            canonical_version=canonical_version,
            reason=str(entry["reason"]).strip(),
            path=project_path,
        )
    return sources


@dataclasses.dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int
    max_tokens: int
    overlap_tokens: int


@dataclasses.dataclass(frozen=True)
class DedupConfig:
    near_duplicate_check: bool
    near_duplicate_threshold: float


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    seed: int
    test_fraction: float
    val_fraction: float


@dataclasses.dataclass(frozen=True)
class QAConfig:
    max_pairs_per_document: int
    llm_model: str
    enable_llm_judge: bool


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    chunking: ChunkingConfig
    dedup: DedupConfig
    split: SplitConfig
    qa: QAConfig


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Lee la clave `pipeline` de config.yaml y la devuelve como dataclasses con
    tipos definidos, no como un diccionario suelto.

    Por que hacerlo asi, en vez de simplemente pasar `raw["chunking"]["target_tokens"]`
    por todo el codigo: si alguien escribe mal el nombre de una clave en el YAML (por
    ejemplo "traget_tokens" en vez de "target_tokens"), el error aparece ahi mismo,
    claro ("falta el argumento target_tokens"), en vez de que el chunker use un valor
    por defecto sin avisar y el problema recien se note al revisar los chunks ya generados.
    """
    raw = load_config_yaml(path)["pipeline"]
    return PipelineConfig(
        chunking=ChunkingConfig(**raw["chunking"]),
        dedup=DedupConfig(**raw["dedup"]),
        split=SplitConfig(**raw["split"]),
        qa=QAConfig(**raw["qa"]),
    )
