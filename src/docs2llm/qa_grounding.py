"""Control de calidad de los pares de pregunta-respuesta, en 3 niveles de
prioridad segun el tiempo disponible (ver notebooks/diagnostico.ipynb, seccion 9):

  1. Chequeo de evidencia obligatorio: la respuesta tiene que venir acompañada de una cita textual EXACTA del chunk
     fuente. Si esa cita no existe literalmente en el chunk, el par se descarta --
     es el mecanismo que mas rinde por el esfuerzo que lleva: convierte el
     "respaldo de la respuesta" (grounding) de algo que "parece confiable" a algo
     verificable con un simple chequeo, sin gastar ni una llamada extra a un
     modelo de lenguaje.
  2. Reglas baratas: longitud razonable de la pregunta/respuesta, que no esten
     vacias, y descartar preguntas casi-duplicadas DENTRO DEL MISMO PROYECTO
     (para que el dataset final no termine con 5 variantes de "que tipo de dato
     tiene X" sobre el mismo campo).
  3. Un modelo como "juez" (extra opcional, se activa con
     config.qa.enable_llm_judge): una segunda llamada a un modelo mas economico
     que evalua si la respuesta esta de verdad respaldada por el chunk. No esta
     activado por defecto -- ver qa_build.py.
"""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass
class QualityChecks:
    evidence_verified: bool
    length_ok: bool
    not_near_duplicate_question: bool
    llm_judge_grounded: bool | None = None
    llm_judge_reasoning: str | None = None

    @property
    def qc_passed(self) -> bool:
        """True si pasan los 3 chequeos basicos; si ademas corrio el modelo como
        juez (el extra opcional), tambien tiene que haber dicho que la respuesta
        esta respaldada."""
        base = self.evidence_verified and self.length_ok and self.not_near_duplicate_question
        if self.llm_judge_grounded is None:
            return base
        return base and self.llm_judge_grounded


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def check_evidence(evidence_span: str, chunk_text: str) -> bool:
    """Confirma que `evidence_span` sea una cita literal (una subcadena exacta,
    despues de juntar los espacios) de `chunk_text` -- el chequeo mas importante
    de todo el dataset de preguntas y respuestas.

    Por que exacto y no un match aproximado: un match aproximado no prueba nada
    real, solo mide que el modelo "confia en si mismo" al responder (ver
    notebooks/diagnostico.ipynb, seccion 9).
    """
    if not evidence_span or not evidence_span.strip():
        return False
    return _collapse_whitespace(evidence_span) in _collapse_whitespace(chunk_text)


# Una pregunta o respuesta mas corta que esto no alcanza a formular algo real
# ("¿Qué es X?" ya son 12 caracteres, asi que el minimo deja margen de sobra);
# mas larga que el maximo, probablemente sea un parrafo pegado por error, no una
# pregunta de verdad.
_MIN_QUESTION_CHARS = 8
_MAX_QUESTION_CHARS = 300


def check_length(question: str, answer: str) -> bool:
    """True si la pregunta mide entre 8 y 300 caracteres y la respuesta no esta
    vacia (ver _MIN_QUESTION_CHARS/_MAX_QUESTION_CHARS arriba para saber por que
    se eligio ese rango)."""
    q, a = question.strip(), answer.strip()
    return bool(a) and _MIN_QUESTION_CHARS <= len(q) <= _MAX_QUESTION_CHARS


def _normalize_question(question: str) -> str:
    # Pasa todo a minuscula, junta espacios, y saca signos de interrogacion o
    # puntuacion del final: dos preguntas que solo se diferencian en "?" vs
    # "¿...?", o en mayusculas/minusculas, tienen que contar como LA MISMA
    # pregunta para el chequeo de casi-duplicados.
    normalized = _collapse_whitespace(question.lower())
    return normalized.strip("¿?.! ")


def find_near_duplicate_question_indices(questions_by_project: list[tuple[str, str]]) -> set[int]:
    """Recibe una lista [(proyecto, pregunta), ...] en el orden en que se van a
    evaluar, y devuelve el conjunto de POSICIONES que hay que rechazar por ser
    casi-duplicadas de una pregunta ya vista ANTES en el mismo proyecto (la
    primera vez que aparece una pregunta, siempre se conserva). La comparacion es
    dentro de un mismo proyecto, no en todo el corpus: dos preguntas identicas
    sobre proyectos distintos (por ejemplo, "¿que sitios soporta?") no son
    duplicados de verdad, son solo una coincidencia de fraseo sobre temas distintos.
    """
    seen_by_project: dict[str, set[str]] = {}
    duplicate_indices: set[int] = set()
    for i, (project, question) in enumerate(questions_by_project):
        normalized = _normalize_question(question)
        seen = seen_by_project.setdefault(project, set())
        if normalized in seen:
            duplicate_indices.add(i)
        else:
            seen.add(normalized)
    return duplicate_indices
