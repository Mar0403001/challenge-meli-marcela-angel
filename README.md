# docs2llm — de documentación cruda a datasets para LLMs

Pipeline local que convierte `docs_raw/` (documentación Markdown desordenada de 10
proyectos de software ficticios) en dos datasets JSONL:

- **`data/processed/corpus.jsonl`** — corpus normalizado por chunks, con procedencia completa. Útil para retrieval/RAG o continued pretraining (CPT).
- **`data/processed/qa.jsonl`** — pares pregunta-respuesta derivados de la documentación, con evidencia verificada. Útil para SFT o evaluación.

Ver **[notebooks/diagnostico.ipynb](notebooks/diagnostico.ipynb)** para la justificación detallada de cada decisión (con comandos y evidencia real ejecutándose en vivo, no solo argumentos).

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

copy .env.example .env              # completar ANTHROPIC_API_KEY (opcional, ver abajo)

python -m docs2llm build-corpus     # genera corpus.jsonl
python -m docs2llm build-qa         # genera qa.jsonl
```

`build-qa` sin `ANTHROPIC_API_KEY` configurada (o con `--no-llm`) igual funciona: genera Q&A solo con los templates deterministas (tablas y specs OpenAPI), avisando por qué faltan los pares de prosa.

Correr los tests:

```powershell
pytest tests/ -v
```

Abrir el notebook de evidencia (mismo `requirements.txt`, ya instalado arriba):

```powershell
jupyter notebook notebooks/diagnostico.ipynb
```

## Estructura de `src/docs2llm/`

Un archivo por etapa, sin subcarpetas — el orden alfabético agrupa cada familia
(`parse_*` = parsing, `qa_*` = generación de Q&A):

```
cli.py             CLI: build-corpus / build-qa / validate-canonical-sources
config.py          carga config/config.yaml en dataclasses tipadas
discovery.py       resuelve carpeta canónica por proyecto + manifiesto de archivos
normalize.py       3 fixes: headings rotos, doble-escape HTML, ruido Mermaid
parse_html.py      limpia HTML embebido a texto plano
parse_markdown.py  AST de Markdown -> árbol Section/Block
parse_openapi.py   swagger.yaml -> Section/Block (paths + schemas)
parse_units.py     dataclasses compartidas Section/Block
chunking.py        bin-packing heading-aware -> Chunk (nunca parte tablas/código)
dedup.py           hash exacto + embeddings (near-duplicates) + componentes conexas entre archivos
splitting.py       asigna train/val/test global por peso en chunks, por componente conexa
corpus_build.py    orquesta discovery->parse->chunk->dedup->split -> corpus.jsonl
qa_templates.py    Q&A determinista desde tablas/OpenAPI
qa_llm_generate.py Q&A vía API de Claude para prosa
qa_grounding.py    control de calidad (evidence/longitud/dedup de pregunta)
qa_units.py        dataclass compartida QAPair
qa_build.py        orquesta templates+LLM -> QC -> tope anti-volumen -> qa.jsonl
```

## Qué hace el pipeline, en 2 etapas

**Etapa A — `build-corpus`**: resuelve qué carpeta de versión usar por proyecto (`config/config.yaml`, clave `canonical_sources` — decisión curada y justificada, no automática — ver notebook, sección 1), parsea Markdown vía AST (`markdown-it-py`) y OpenAPI vía parser dedicado, normaliza (arregla encabezados rotos, corrige un bug de doble-escape HTML, colapsa ruido de diagramas Mermaid), arma chunks heading-aware sin partir tablas/código, deduplica en dos capas —hash exacto a nivel de bloque (no de chunk ya empaquetado — ver notebook, sección 6) y embeddings semánticos para duplicados aproximados (contenido que cambió de texto pero no de significado)—, y asigna train/val/test agrupando por documentos que comparten contenido duplicado, exacto o aproximado (ver notebook, sección 7), para que ningún fragmento repetido quede en dos splits a la vez.

**Etapa B — `build-qa`**: lee `corpus.jsonl` (nunca recalcula el split). Genera candidatos con dos métodos —templates deterministas sobre tablas/OpenAPI (grounding garantizado por construcción) y la API de Claude sobre prosa/reglas de negocio (con `evidence_span` obligatorio)—, valida cada candidato (evidencia verificada como subcadena literal, longitud razonable, sin preguntas casi-duplicadas), y aplica un tope anti-volumen por documento para que un archivo con muchas secciones no domine el dataset.

## Esquema de las salidas

**`corpus.jsonl`**: `doc_id, id, project, project_version, source_path, source_url_hint, section_path, heading, chunk_index, n_chunks_in_doc, char_start, char_end, text, n_tokens, language, content_type, is_low_signal, content_hash_sha256, duplicate_of, split`

**`qa.jsonl`**: `id, project, project_version, source_path, section_path, source_chunk_id, question, answer, answer_type, evidence_span, generation_method, generation_detail, quality_checks, qc_passed, split`

Ambas trazan cada fila hasta su origen exacto (`source_path` + `section_path`), y `qa.source_chunk_id` referencia directamente la fila de `corpus.jsonl` de la que salió.

## Decisiones clave (resumen — detalle completo en `notebooks/diagnostico.ipynb`)

- **"latest" no es confiable como señal de versión más reciente**: se verificó con hashes que en 1 de los 4 proyectos con múltiples versiones, "latest" era la copia más *vieja*. Se optó por una tabla de curación explícita con la razón de cada elección, no una heurística automática.
- **Deduplicación en dos capas**: exacta por hash (a nivel de bloque atómico, no de chunk ya empaquetado — un fragmento repetido que queda mezclado con contexto distinto en cada archivo no se detecta si se compara el chunk final) y aproximada por embeddings (`sentence-transformers`, modelo `all-MiniLM-L6-v2` vendorizado en `vendor/sentence_transformers_cache/`, mismo patrón que `tiktoken`) para contenido que cambia de texto pero no de significado — el caso real es `2p-revenue-optimizer-api`, donde una tabla tiene columnas renombradas entre versiones (`created_date`→`date`). El umbral de similitud (0.9) se validó contra el corpus real: los chunks `api_spec` quedan excluidos de esta capa porque su formato fijo (ver `parse_openapi.py`) hace que operaciones genuinamente distintas den similitud alta solo por compartir plantilla, no contenido repetido. Ambas capas alimentan las mismas componentes conexas entre archivos, para que el split no separe contenido duplicado (exacto o aproximado) entre train y test.
- **Split 60/20/20 por cantidad de CHUNKS, global sobre todo el corpus** (no por cantidad de documentos, no recalculado por proyecto): los documentos de este corpus varían mucho en tamaño (de 1 a más de 50 chunks), así que repartir "por cabeza" no ve ese peso — un documento de 51 chunks cuenta igual que uno de 1 chunk. La solución es un algoritmo LPT (longest-processing-time-first, un heurístico clásico de balanceo de carga) que corre sobre TODOS los grupos de documentos del corpus a la vez: ordena por peso (en chunks) de mayor a menor, y cada uno se lo lleva el split que en ese momento esté más lejos de su cuota — con el estado compartido entre proyectos, el resultado real termina en 59.9/20.1/20. El trade off de este enfoque: no se garantiza que cada proyecto tenga representación en los 3 splits, queda auditado con warnings explícitos en `split_report.json` cuando un proyecto pierde representación en val o test.
- **Reintentos + aislamiento de errores por chunk en `build-qa`**: cada chunk de prosa es una llamada separada a la API de Claude (~250 en este corpus), y las filas finales solo se escriben a disco al terminar todo el lote. `qa_llm_generate.py` reintenta con espera creciente los errores transitorios (rate limit, 5xx) — no los que no se arreglan reintentando (request mal formado, auth). Si un chunk puntual falla igual (agotó los reintentos, o el error no era reintentable), `qa_build.py` lo trata como si no hubiera producido candidatos y sigue con el resto, en vez de perder el trabajo ya hecho sobre los demás ~250 chunks — queda contado en `chunks_con_error_llm` del reporte final.
- **Q&A híbrido**: templates para contenido estructurado (grounding gratis) + LLM para prosa (grounding verificado con cita textual obligatoria).
- **`tiktoken` con encoding vendorizado**: se usa el tokenizador real (`cl100k_base`) para dimensionar chunks. La primera vez que se probó, la descarga del archivo de encoding falló por un proxy de inspección TLS (ver notebook, sección 5) y se reemplazó por una aproximación de caracteres/token; luego se identificó que el mismo fix ya usado para la API de Claude (`truststore`) también resuelve esa descarga, y se optó por ir un paso más allá: el archivo de encoding (~1.7MB) quedó vendorizado en `vendor/tiktoken_cache/`, así el conteo de tokens es exacto y el pipeline sigue sin depender de la red en ningún punto. Sin `truststore`: la API de Claude fallaba por un problema de certificados TLS en redes con proxy de inspección (ver notebook, sección 8).

## Limitaciones conocidas

- La detección de idioma es heurística (conteo de stopwords), no un clasificador robusto.
- Los templates de OpenAPI describen solo 1 nivel de propiedades anidadas en schemas complejos.

## Archivo de presentación

- Dentro de la carpeta `presentation/` se encuentra el archivo
  `presentation-pipeline-flow.html`, utilizado como apoyo para la sustentación de la solución.

- La presentación documenta de forma visual y detallada:
  - El contexto del desafío y las características del corpus procesado.
  - La arquitectura completa del pipeline (`build-corpus` y `build-qa`).
  - Las decisiones de diseño y los criterios técnicos adoptados en cada etapa.
  - Los hallazgos encontrados durante el análisis y procesamiento de la documentación.
  - Los mecanismos implementados para garantizar calidad, reproducibilidad y evitar data leakage.
  - Los trade-offs considerados y las limitaciones conocidas de la solución.
  - La estructura y estadísticas de los datasets finales generados (`corpus.jsonl` y `qa.jsonl`).

- El objetivo del documento es facilitar la comprensión del razonamiento detrás de la implementación, complementando el código fuente y permitiendo seguir el flujo completo desde los documentos originales hasta la generación del dataset final para LLMs.


