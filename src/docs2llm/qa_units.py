"""Esta clase la comparten qa_templates.py y qa_llm_generate.py -- las dos formas de
generar una pregunta (con reglas fijas, o con un modelo de lenguaje) devuelven el
mismo tipo de par, para que qa_build.py les pueda aplicar el mismo control de
calidad y el mismo tope de pares por documento, sin importar de dónde vino cada uno.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class QAPair:
    question: str
    answer: str
    # Cita textual EXACTA (una subcadena literal del fragmento fuente) que sustenta
    # la respuesta -- qa_grounding.py verifica despues que sea real, no inventada.
    evidence_span: str
    answer_type: str  # "extractive" | "generative" | "numeric" | "boolean" | "enum"
    generation_detail: str  # que plantilla puntual, o que modelo de LLM, la genero
