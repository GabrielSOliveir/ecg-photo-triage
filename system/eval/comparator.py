"""Comparador estruturado: ECGFindings (sistema) vs ECGFindings (GT).

Produz ComparisonResult com 1 item por campo clinicamente relevante.

Categorias:
  AGREE             — concordância
  CATCH             — sistema viu, GT não — candidato a achado novo (precisa revisão médica)
  MISS              — GT viu, sistema não
  HALLUCINATION     — sistema afirma X, GT afirma Y (clinicamente conflitante)
  NEUTRAL_DISAGREE  — discordância clinicamente irrelevante

Premissa importante: GT vem de texto telegráfico parseado. Ausência no GT ≠ ausência
no exame — pode ser que a médica simplesmente não escreveu. Por isso, "sistema viu,
GT não" é CATCH (suspeito) e não automaticamente HALLUCINATION."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import ECGFindings, ComparisonResult, ComparisonItem


def _set_str(items: list[str]) -> str:
    if not items:
        return "—"
    return ", ".join(items)


def _compare_ritmo(s: ECGFindings, g: ECGFindings) -> list[ComparisonItem]:
    if s.ritmo == "indeterminado" and g.ritmo == "indeterminado":
        return []
    if s.ritmo == g.ritmo:
        return [ComparisonItem(campo="ritmo", categoria="AGREE",
                                sistema=s.ritmo, gt=g.ritmo, peso_clinico="alto")]
    if g.ritmo == "indeterminado":
        return [ComparisonItem(campo="ritmo", categoria="CATCH",
                                sistema=s.ritmo, gt="—", peso_clinico="alto")]
    if s.ritmo == "indeterminado":
        return [ComparisonItem(campo="ritmo", categoria="MISS",
                                sistema="—", gt=g.ritmo, peso_clinico="alto")]
    return [
        ComparisonItem(campo="ritmo", categoria="HALLUCINATION",
                       sistema=s.ritmo, gt=g.ritmo, peso_clinico="alto"),
    ]


def _compare_fc(s: ECGFindings, g: ECGFindings) -> list[ComparisonItem]:
    if not s.fc_bpm and not g.fc_bpm:
        return []
    if not g.fc_bpm:
        # GT não escreveu FC explicitamente — não dá pra avaliar
        return []
    if not s.fc_bpm:
        return [ComparisonItem(campo="fc_bpm", categoria="MISS",
                                sistema="—", gt=str(g.fc_bpm), peso_clinico="baixo")]
    diff = abs(s.fc_bpm - g.fc_bpm)
    if diff <= 10:
        return [ComparisonItem(campo="fc_bpm", categoria="AGREE",
                                sistema=str(s.fc_bpm), gt=str(g.fc_bpm), peso_clinico="baixo")]
    if diff <= 25:
        return [ComparisonItem(campo="fc_bpm", categoria="NEUTRAL_DISAGREE",
                                sistema=str(s.fc_bpm), gt=str(g.fc_bpm), peso_clinico="baixo")]
    # Diff grande
    return [ComparisonItem(campo="fc_bpm", categoria="HALLUCINATION",
                            sistema=str(s.fc_bpm), gt=str(g.fc_bpm), peso_clinico="medio")]


def _compare_set_field(
    campo: str,
    sistema_items: list[str],
    gt_items: list[str],
    peso: str = "medio",
) -> list[ComparisonItem]:
    """Compara campos que são listas/sets. Cada elemento que só está em um lado
    vira CATCH ou MISS. Interseção vira AGREE (um único item)."""
    s_set, g_set = set(sistema_items), set(gt_items)

    if not s_set and not g_set:
        return []

    items: list[ComparisonItem] = []
    intersec = s_set & g_set
    only_s = s_set - g_set
    only_g = g_set - s_set

    if intersec:
        items.append(ComparisonItem(
            campo=campo, categoria="AGREE",
            sistema=_set_str(sorted(intersec)),
            gt=_set_str(sorted(intersec)),
            peso_clinico=peso,
        ))
    for x in sorted(only_s):
        items.append(ComparisonItem(
            campo=campo, categoria="CATCH",
            sistema=x, gt="—", peso_clinico=peso,
        ))
    for x in sorted(only_g):
        items.append(ComparisonItem(
            campo=campo, categoria="MISS",
            sistema="—", gt=x, peso_clinico=peso,
        ))
    return items


def _compare_bool(
    campo: str,
    sistema: bool,
    gt: bool,
    peso: str = "medio",
) -> list[ComparisonItem]:
    if not sistema and not gt:
        return []
    if sistema and gt:
        return [ComparisonItem(campo=campo, categoria="AGREE",
                                sistema="sim", gt="sim", peso_clinico=peso)]
    if sistema and not gt:
        return [ComparisonItem(campo=campo, categoria="CATCH",
                                sistema="sim", gt="—", peso_clinico=peso)]
    return [ComparisonItem(campo=campo, categoria="MISS",
                            sistema="—", gt="sim", peso_clinico=peso)]


def _compare_intervalos(s: ECGFindings, g: ECGFindings) -> list[ComparisonItem]:
    items: list[ComparisonItem] = []
    # QTc: relevante se ultrapassa thresholds clínicos
    if s.qtc_ms and g.qtc_ms:
        s_long = s.qtc_ms >= 460
        g_long = g.qtc_ms >= 460
        if s_long == g_long:
            items.append(ComparisonItem(
                campo="qtc", categoria="AGREE",
                sistema=f"{s.qtc_ms}ms", gt=f"{g.qtc_ms}ms",
                peso_clinico="medio" if s_long else "baixo",
            ))
        elif s_long:
            items.append(ComparisonItem(
                campo="qtc", categoria="CATCH",
                sistema=f"{s.qtc_ms}ms (longo)", gt=f"{g.qtc_ms}ms",
                peso_clinico="medio",
            ))
        else:
            items.append(ComparisonItem(
                campo="qtc", categoria="MISS",
                sistema=f"{s.qtc_ms}ms", gt=f"{g.qtc_ms}ms (longo)",
                peso_clinico="medio",
            ))
    elif s.qtc_ms and s.qtc_ms >= 460 and not g.qtc_ms:
        items.append(ComparisonItem(
            campo="qtc", categoria="CATCH",
            sistema=f"{s.qtc_ms}ms (longo)", gt="—",
            peso_clinico="medio",
        ))
    return items


def _compare_st(
    campo: str,
    sistema_derivs: list[str],
    sistema_mag: float,
    gt_derivs: list[str],
    gt_mag: float,
    peso: str = "alto",
) -> list[ComparisonItem]:
    return _compare_set_field(campo, sistema_derivs, gt_derivs, peso=peso)


def _compare_onda_t(s: ECGFindings, g: ECGFindings) -> list[ComparisonItem]:
    # Reduz a strings tipo "invertida:V1,V2"
    def _flatten(alt_list):
        out = []
        for a in alt_list:
            d = ",".join(a.derivacoes) if a.derivacoes else "difuso"
            out.append(f"{a.tipo}:{d}")
        return out

    return _compare_set_field("onda_t", _flatten(s.onda_t_alteracoes), _flatten(g.onda_t_alteracoes), peso="medio")


def compare(sistema: ECGFindings, gt: ECGFindings, case_id: int) -> ComparisonResult:
    items: list[ComparisonItem] = []

    items += _compare_ritmo(sistema, gt)
    items += _compare_fc(sistema, gt)

    items += _compare_set_field("bloqueios",
                                 sorted(sistema.bloqueios), sorted(gt.bloqueios),
                                 peso="alto")
    items += _compare_set_field("areas_inativas",
                                 sorted(sistema.areas_inativas), sorted(gt.areas_inativas),
                                 peso="alto")
    items += _compare_set_field("achados_especiais",
                                 sorted(sistema.achados_especiais),
                                 sorted(gt.achados_especiais),
                                 peso="alto")

    items += _compare_st("st_supra",
                         sistema.st_supra.derivacoes, sistema.st_supra.magnitude_mm_max,
                         gt.st_supra.derivacoes, gt.st_supra.magnitude_mm_max,
                         peso="alto")
    items += _compare_st("st_infra",
                         sistema.st_infra.derivacoes, sistema.st_infra.magnitude_mm_max,
                         gt.st_infra.derivacoes, gt.st_infra.magnitude_mm_max,
                         peso="alto")

    items += _compare_onda_t(sistema, gt)

    items += _compare_bool("pre_excitacao", sistema.pre_excitacao, gt.pre_excitacao, peso="alto")
    items += _compare_bool("marcapasso", sistema.marcapasso, gt.marcapasso, peso="alto")
    items += _compare_bool("hipertrofia_ve", sistema.hipertrofia_ve, gt.hipertrofia_ve, peso="medio")
    items += _compare_bool("hipertrofia_vd", sistema.hipertrofia_vd, gt.hipertrofia_vd, peso="medio")
    items += _compare_bool("sobrecarga_atrial_esq", sistema.sobrecarga_atrial_esq, gt.sobrecarga_atrial_esq, peso="baixo")
    items += _compare_bool("sobrecarga_atrial_dir", sistema.sobrecarga_atrial_dir, gt.sobrecarga_atrial_dir, peso="baixo")

    items += _compare_intervalos(sistema, gt)

    return ComparisonResult(case_id=case_id, itens=items)
