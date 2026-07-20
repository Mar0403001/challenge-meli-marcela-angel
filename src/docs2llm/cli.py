"""Punto de entrada del pipeline: `python -m docs2llm <subcomando>`.

Se uso argparse (viene incluido en Python, no hay que instalar nada aparte) en
vez de una libreria como typer/click: son solo 2-3 subcomandos, asi que no se
justifica agregar una dependencia extra solo para leer argumentos de la terminal.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# truststore hace que Python confie en los certificados de seguridad usando el
# almacen del sistema operativo, en vez del que trae por su cuenta la libreria
# `certifi`. Hace falta para que las llamadas a la API de Claude
# (qa_llm_generate.py) funcionen detras de un proxy corporativo que inspecciona el
# trafico. Se activa ANTES de importar anthropic/httpx (esas librerias tienen que
# armar sus conexiones seguras DESPUES de este ajuste, no antes). Ver
# requirements.txt y notebooks/diagnostico.ipynb (seccion 8) para el diagnostico
# completo: se detecto esto porque build-qa fallaba con
# "CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate", a pesar de
# que `curl` (que sí usa el almacen del sistema operativo) llegaba sin problema al
# mismo sitio.
import truststore

truststore.inject_into_ssl()

# python-dotenv lee el archivo .env local y carga sus variables (como
# ANTHROPIC_API_KEY) al entorno, para no tener que configurarlas a mano en cada
# sesion de terminal.
from dotenv import load_dotenv

from docs2llm.config import REPO_ROOT, load_canonical_sources, load_pipeline_config
from docs2llm.corpus_build import CorpusRow, build_corpus_rows
from docs2llm.qa_build import build_qa_rows

_DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"

# Carga el archivo .env si existe (con la variable ANTHROPIC_API_KEY) -- no falla
# si el archivo no existe (por ejemplo, si solo se esta corriendo build-corpus,
# que no necesita esa clave). Se hace al importar este archivo, no dentro de
# main(), para que tambien funcione si alguien llama a las funciones de cli.py
# directamente desde un script o un notebook, sin pasar por la terminal.
load_dotenv(REPO_ROOT / ".env")


def _write_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # encoding="utf-8" puesto explicitamente, SIEMPRE: en Windows, por defecto,
    # open() usaria la pagina de codigos local (cp1252) para escribir el archivo,
    # lo cual arruinaria de verdad (no solo en pantalla) cualquier tilde o "ñ" del
    # contenido en español si no se indicara esto (ver
    # notebooks/diagnostico.ipynb, seccion 4, la nota tecnica sobre cp1252).
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dataclasses.asdict(row), ensure_ascii=False) + "\n")


def _write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_build_corpus(args: argparse.Namespace) -> None:
    """Subcomando `build-corpus`: corre el pipeline completo de corpus_build.py y
    escribe corpus.jsonl + manifest.json + dedup_report.json + near_dedup_report.json
    + split_report.json."""
    sources = load_canonical_sources()
    pipeline_config = load_pipeline_config()
    rows, split_report, near_duplicate_pairs = build_corpus_rows(sources, pipeline_config)

    out_dir = Path(args.out_dir)
    _write_jsonl(rows, out_dir / "corpus.jsonl")
    _write_json(split_report, out_dir / "split_report.json")

    duplicate_rows = [r for r in rows if r.duplicate_of is not None]
    dedup_report = {
        "total_chunks": len(rows),
        "chunks_marcados_como_duplicado": len(duplicate_rows),
        "clusters_de_duplicados_exactos": len({r.duplicate_of for r in duplicate_rows}),
        "ejemplos": [
            {
                "id": r.id,
                "duplicate_of": r.duplicate_of,
                "source_path": r.source_path,
                "section_path": r.section_path,
            }
            for r in duplicate_rows[:20]
        ],
    }
    _write_json(dedup_report, out_dir / "dedup_report.json")

    rows_by_id = {r.id: r for r in rows}
    near_dedup_report = {
        "near_duplicate_check_activo": pipeline_config.dedup.near_duplicate_check,
        "near_duplicate_threshold": pipeline_config.dedup.near_duplicate_threshold,
        "pares_encontrados": len(near_duplicate_pairs),
        "ejemplos": [
            {
                "a": id_a,
                "b": id_b,
                "source_path_a": rows_by_id[id_a].source_path,
                "source_path_b": rows_by_id[id_b].source_path,
            }
            for id_a, id_b in near_duplicate_pairs[:20]
        ],
    }
    _write_json(near_dedup_report, out_dir / "near_dedup_report.json")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_proyectos": len(sources),
        "total_chunks": len(rows),
        "chunks_por_split": dict(Counter(r.split for r in rows)),
        "chunks_por_content_type": dict(Counter(r.content_type for r in rows)),
        "chunks_is_low_signal": sum(1 for r in rows if r.is_low_signal),
        "canonical_sources_usadas": {p: s.canonical_version for p, s in sources.items()},
    }
    _write_json(manifest, out_dir / "manifest.json")

    print(f"corpus.jsonl: {len(rows)} filas -> {out_dir / 'corpus.jsonl'}")
    print(f"  por split: {manifest['chunks_por_split']}")
    print(f"  por content_type: {manifest['chunks_por_content_type']}")
    print(f"  is_low_signal: {manifest['chunks_is_low_signal']}")
    print(f"  duplicados marcados: {dedup_report['chunks_marcados_como_duplicado']}")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_validate_canonical_sources(args: argparse.Namespace) -> None:
    """Extra de bajo costo (ver notebooks/diagnostico.ipynb, seccion 1): vuelve a
    calcular los hashes de TODAS las carpetas de version de cada proyecto (no
    solo la version elegida) y los compara contra la version canonica elegida en
    config/config.yaml (clave `canonical_sources`) -- asi, si docs_raw/ cambia en
    el futuro (nuevas versiones, contenido vuelto a exportar), queda claro si el
    argumento que se documento en el `reason` de cada proyecto sigue siendo
    cierto. Es solo informativo: NO hace fallar el build, solo imprime un
    diagnostico (el mismo que se hizo a mano al escribir esa configuracion la
    primera vez, pero ahora reproducible con un comando).
    """
    from docs2llm.config import DOCS_RAW_DIR, load_config_yaml

    raw = load_config_yaml()["canonical_sources"]

    for project, entry in sorted(raw["projects"].items()):
        project_dir = DOCS_RAW_DIR / project
        version_dirs = sorted(d for d in project_dir.iterdir() if d.is_dir())
        if len(version_dirs) < 2:
            continue  # no hay nada que comparar, solo hay una carpeta de version disponible

        canonical_version = str(entry["canonical"])
        canonical_dir = project_dir / canonical_version
        canonical_hashes = {f.relative_to(canonical_dir).as_posix(): _sha256_file(f) for f in canonical_dir.rglob("*") if f.is_file()}

        print(f"\n{project} (canonica configurada: {canonical_version})")
        for version_dir in version_dirs:
            if version_dir.name == canonical_version:
                continue
            other_hashes = {f.relative_to(version_dir).as_posix(): _sha256_file(f) for f in version_dir.rglob("*") if f.is_file()}
            common = set(canonical_hashes) & set(other_hashes)
            different = sorted(p for p in common if canonical_hashes[p] != other_hashes[p])
            only_canonical = sorted(set(canonical_hashes) - set(other_hashes))
            only_other = sorted(set(other_hashes) - set(canonical_hashes))

            if not different and not only_canonical and not only_other:
                print(f"  vs {version_dir.name}: identico byte a byte ({len(common)} archivos)")
            else:
                print(
                    f"  vs {version_dir.name}: {len(different)} archivo(s) distinto(s), "
                    f"{len(only_canonical)} solo en canonica, {len(only_other)} solo en '{version_dir.name}'"
                )
                for p in different[:5]:
                    print(f"    - distinto: {p}")

    print("\n(Esto es informativo: compara el estado actual de docs_raw/ contra el 'reason' documentado en config/config.yaml -- no falla el build.)")


def _build_anthropic_client_or_none():
    """Intenta crear el cliente de Claude; si falta ANTHROPIC_API_KEY, devuelve
    None en vez de romper -- build-qa sigue funcionando solo con plantillas
    (tablas/OpenAPI), y se avisa con claridad que la parte que usa un modelo de
    lenguaje (la prosa) se salteo."""
    import anthropic

    try:
        return anthropic.Anthropic()
    except anthropic.AnthropicError:
        print(
            "AVISO: no se encontro ANTHROPIC_API_KEY (ver .env.example). Se genera "
            "Q&A solo con templates deterministicos (tablas/OpenAPI); los chunks de "
            "prosa no producen preguntas en esta corrida.",
            file=sys.stderr,
        )
        return None


def cmd_build_qa(args: argparse.Namespace) -> None:
    """Subcomando `build-qa`: lee corpus.jsonl (hace falta haber corrido
    build-corpus antes) y escribe qa.jsonl + qc_report.json usando
    qa_build.build_qa_rows."""
    pipeline_config = load_pipeline_config()
    out_dir = Path(args.out_dir)

    corpus_path = out_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"No existe {corpus_path}. Corre primero: python -m docs2llm build-corpus")

    corpus_rows = [
        CorpusRow(**json.loads(line)) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    llm_client = None if args.no_llm else _build_anthropic_client_or_none()
    qa_rows, qc_report = build_qa_rows(corpus_rows, pipeline_config.qa, llm_client=llm_client)

    _write_jsonl(qa_rows, out_dir / "qa.jsonl")
    _write_json(qc_report, out_dir / "qc_report.json")

    print(f"qa.jsonl: {len(qa_rows)} filas -> {out_dir / 'qa.jsonl'}")
    print(f"  candidatos generados: {qc_report['candidatos_generados']} ({qc_report['candidatos_por_metodo']})")
    print(f"  pasaron QC: {qc_report['candidatos_que_pasaron_qc']}")
    print(f"  filas finales (tras tope anti-volumen): {qc_report['filas_finales_tras_tope_anti_volumen']}")
    print(f"  por split: {qc_report['filas_por_split']}")
    print(f"  por metodo: {qc_report['filas_por_metodo']}")
    if qc_report["chunks_con_error_llm"]:
        print(
            f"  AVISO: {qc_report['chunks_con_error_llm']} chunk(s) de prosa fallaron al "
            f"generar Q&A vía LLM y se saltearon (ver stderr arriba para el detalle) -- "
            f"el resto de la corrida siguió normalmente."
        )


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de `python -m docs2llm`: registra los subcomandos
    (build-corpus, build-qa, validate-canonical-sources) y llama al que
    corresponda."""
    # Fuerza que la salida de la terminal use UTF-8: en Windows, sin esto, un
    # print() de texto con tildes/ñ se ve corrupto en la consola por defecto
    # (cp1252) -- ver notebooks/diagnostico.ipynb, seccion 4. Esto SOLO afecta lo
    # que se imprime en pantalla, nunca a los archivos que se escriben (esos
    # siempre usan encoding="utf-8" explicito, ver _write_jsonl/_write_json).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="docs2llm", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_corpus_parser = subparsers.add_parser("build-corpus", help="Genera corpus.jsonl a partir de docs_raw/")
    build_corpus_parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR), help="Carpeta de salida (default: data/processed)")
    build_corpus_parser.set_defaults(func=cmd_build_corpus)

    build_qa_parser = subparsers.add_parser("build-qa", help="Genera qa.jsonl a partir de corpus.jsonl (requiere haber corrido build-corpus antes)")
    build_qa_parser.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR), help="Carpeta de entrada/salida (default: data/processed)")
    build_qa_parser.add_argument(
        "--no-llm", action="store_true",
        help="No llamar a la API de Claude: genera Q&A solo con templates deterministicos (tablas/OpenAPI)",
    )
    build_qa_parser.set_defaults(func=cmd_build_qa)

    validate_parser = subparsers.add_parser(
        "validate-canonical-sources",
        help="Recalcula hashes de docs_raw/ y compara contra config/config.yaml (informativo, no falla)",
    )
    validate_parser.set_defaults(func=cmd_validate_canonical_sources)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
