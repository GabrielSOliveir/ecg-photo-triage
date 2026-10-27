"""Alertas e piso de risco derivados de sinais vitais.
Determinístico, sem LLM. O resultado *sempre* sobrepõe a inferência do modelo."""

from typing import NamedTuple

from schemas import PatientData, Risco


class VitalAlert(NamedTuple):
    motivo: str
    piso_risco: Risco


def vital_floor(patient: PatientData) -> tuple[Risco, list[VitalAlert]]:
    """Retorna (risco mínimo a impor, lista de alertas)."""
    alertas: list[VitalAlert] = []
    sv = patient.sinais_vitais

    if sv.pa_sistolica is not None:
        if sv.pa_sistolica <= 70:
            alertas.append(VitalAlert(f"PA sistólica {sv.pa_sistolica} — choque", "EMERGENCIA"))
        elif sv.pa_sistolica < 90:
            alertas.append(VitalAlert(f"PA sistólica {sv.pa_sistolica} — hipotensão", "ALTO"))
        elif sv.pa_sistolica >= 220:
            alertas.append(VitalAlert(f"PA sistólica {sv.pa_sistolica} — emergência hipertensiva", "ALTO"))
        elif sv.pa_sistolica >= 180:
            alertas.append(VitalAlert(f"PA sistólica {sv.pa_sistolica} — hipertensão grave", "INTERMEDIARIO"))

    if sv.pa_diastolica is not None and sv.pa_diastolica >= 120:
        alertas.append(VitalAlert(f"PA diastólica {sv.pa_diastolica} — emergência hipertensiva", "ALTO"))

    if sv.fc is not None:
        if sv.fc < 35 or sv.fc > 180:
            alertas.append(VitalAlert(f"FC {sv.fc}", "EMERGENCIA"))
        elif sv.fc < 45 or sv.fc > 150:
            alertas.append(VitalAlert(f"FC {sv.fc}", "ALTO"))
        elif sv.fc < 50 or sv.fc > 120:
            alertas.append(VitalAlert(f"FC {sv.fc}", "INTERMEDIARIO"))

    if sv.sat is not None:
        if sv.sat < 85:
            alertas.append(VitalAlert(f"Sat {sv.sat}% — hipoxemia grave", "EMERGENCIA"))
        elif sv.sat < 90:
            alertas.append(VitalAlert(f"Sat {sv.sat}% — hipoxemia", "ALTO"))
        elif sv.sat < 94:
            alertas.append(VitalAlert(f"Sat {sv.sat}%", "INTERMEDIARIO"))

    if sv.fr is not None:
        if sv.fr < 8 or sv.fr >= 30:
            alertas.append(VitalAlert(f"FR {sv.fr}", "ALTO"))
        elif sv.fr >= 24:
            alertas.append(VitalAlert(f"FR {sv.fr} — taquipneia", "INTERMEDIARIO"))

    piso = max_risco([a.piso_risco for a in alertas]) if alertas else "INDETERMINADO"
    return piso, alertas


_ORDER: list[Risco] = ["INDETERMINADO", "BAIXO", "INTERMEDIARIO", "ALTO", "EMERGENCIA"]


def max_risco(riscos: list[Risco]) -> Risco:
    if not riscos:
        return "INDETERMINADO"
    return max(riscos, key=_ORDER.index)


SYMPTOM_KEYWORDS_CRITICOS = (
    "dor torácica", "dor precordial", "dor no peito", "precordialgia",
    "dispneia", "falta de ar",
    "síncope", "sincope", "desmaio",
    "palpitação", "palpitacao", "palpitações",
    "irradia", "sudorese", "náusea", "nausea",
    "tontura", "lipotimia",
)


def queixa_critica(queixa: str) -> bool:
    if not queixa:
        return False
    q = queixa.lower()
    return any(k in q for k in SYMPTOM_KEYWORDS_CRITICOS)
