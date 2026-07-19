"""Tests de qa_grounding.py: el gate de calidad mas importante del dataset de Q&A."""

from __future__ import annotations

from docs2llm.qa_grounding import (
    QualityChecks,
    check_evidence,
    check_length,
    find_near_duplicate_question_indices,
)


def test_evidence_exacta_pasa():
    chunk = "El limite es de 1000 solicitudes cada 5 minutos para paises RetailHub."
    assert check_evidence("1000 solicitudes cada 5 minutos", chunk) is True


def test_evidence_con_espaciado_distinto_pasa_por_normalizacion():
    """La respuesta puede citar el chunk con saltos de linea/espacios distintos
    (ej. una tabla reformateada) -- se colapsa whitespace antes de comparar, pero
    segue siendo un match EXACTO de las palabras, no fuzzy."""
    chunk = "campo:   valor\ncon    espacios raros"
    assert check_evidence("campo: valor con espacios raros", chunk) is True


def test_evidence_alucinada_no_pasa():
    chunk = "El limite es de 1000 solicitudes cada 5 minutos."
    assert check_evidence("el limite es de 5000 solicitudes cada minuto", chunk) is False


def test_evidence_vacia_no_pasa():
    assert check_evidence("", "cualquier texto") is False
    assert check_evidence("   ", "cualquier texto") is False


def test_longitud_razonable_pasa():
    assert check_length("¿Qué tipo de dato tiene el campo X?", "bigint") is True


def test_pregunta_muy_corta_no_pasa():
    assert check_length("¿Qué?", "algo") is False  # 5 caracteres, bajo el minimo


def test_respuesta_vacia_no_pasa():
    assert check_length("¿Qué tipo de dato tiene el campo X?", "") is False
    assert check_length("¿Qué tipo de dato tiene el campo X?", "   ") is False


def test_preguntas_casi_identicas_mismo_proyecto_se_marcan_duplicadas():
    preguntas = [
        ("proyecto_a", "¿Qué tipo de dato tiene el campo id?"),
        ("proyecto_a", "¿qué tipo de dato tiene el campo id?"),  # solo cambia mayuscula/minuscula
        ("proyecto_a", "¿Qué tipo de dato tiene el campo   id?"),  # espaciado distinto
        ("proyecto_a", "¿Qué tipo de dato tiene el campo name?"),  # pregunta genuinamente distinta
    ]
    duplicadas = find_near_duplicate_question_indices(preguntas)
    assert duplicadas == {1, 2}, "la 1ra aparicion se conserva; la 2da y 3ra son casi-duplicados de la 1ra"


def test_misma_pregunta_en_proyectos_distintos_no_se_marca_duplicada():
    """La misma pregunta sobre DOS proyectos distintos no es un duplicado real --
    son coincidencia de fraseo sobre dominios distintos (ej. '¿que sitios soporta?'
    puede aplicar tanto a price-engine-api como a traffic-gate-api)."""
    preguntas = [
        ("proyecto_a", "¿Qué sitios soporta este endpoint?"),
        ("proyecto_b", "¿Qué sitios soporta este endpoint?"),
    ]
    assert find_near_duplicate_question_indices(preguntas) == set()


def test_quality_checks_qc_passed_requiere_los_3_basicos():
    ok = QualityChecks(evidence_verified=True, length_ok=True, not_near_duplicate_question=True)
    assert ok.qc_passed is True

    falla_evidencia = QualityChecks(evidence_verified=False, length_ok=True, not_near_duplicate_question=True)
    assert falla_evidencia.qc_passed is False


def test_quality_checks_con_llm_judge_negativo_hace_fallar_qc_pese_a_lo_demas_ok():
    con_judge_negativo = QualityChecks(
        evidence_verified=True, length_ok=True, not_near_duplicate_question=True,
        llm_judge_grounded=False, llm_judge_reasoning="la respuesta no se sustenta en el chunk",
    )
    assert con_judge_negativo.qc_passed is False
