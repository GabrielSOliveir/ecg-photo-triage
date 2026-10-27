"""Roda o pipeline sobre os 54 eval_cases, salva runs + parses GT + compara.

Uso:
  python -m eval.runner                # roda tudo (skip já feitos)
  python -m eval.runner --force         # reroda tudo
  python -m eval.runner --only 1,2,5    # apenas IDs específicos

Cuidados:
- Sem few-shot. Sem leak.
- GT parse cacheado por case_id (não reparse se já existir, exceto com --reparse-gt).
- Cada rodada cria um run novo + comparison nova (mantém histórico)."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import PatientData, ECGFindings
from pipeline.orchestrator import analyze
from gt.parser import parse_gt_laudo, GT_PARSER_VERSION, GTParseError
from eval.comparator import compare
from storage import db


def _patient_from_caso_texto(caso_texto: str) -> PatientData:
    """Por enquanto, joga o texto livre no campo 'queixa'. O vision extractor
    e o conduta consomem isso como contexto. Schema completo será populado
    quando houver entrada estruturada (bot Telegram, formulário, etc)."""
    return PatientData(queixa=caso_texto, contexto=caso_texto)


def _ensure_gt_findings(case_id: int, laudo_gt: str, reparse: bool = False) -> ECGFindings:
    if not reparse:
        cached = db.get_gt_findings(case_id)
        if cached:
            return ECGFindings.model_validate_json(cached)
    parsed = parse_gt_laudo(laudo_gt)
    db.save_gt_findings(case_id, parsed.model_dump_json(), GT_PARSER_VERSION)
    return parsed


def run_one(case_row, reparse_gt: bool, force_run: bool,
            vision_model: str | None = None, conduta_model: str | None = None) -> dict:
    cid = case_row["id"]
    print(f"\n── Caso {cid} ──")
    print(f"   Paciente: {case_row['caso_texto'][:80]}...")
    print(f"   GT laudo: {case_row['laudo_gt'][:80]}")

    # Já tem run? skip a menos que --force
    if not force_run and db.latest_run_for("eval", cid):
        print("   [skip] já tem run; use --force para rerodar")
        return {"skipped": True}

    patient = _patient_from_caso_texto(case_row["caso_texto"])
    img_path = Path(case_row["imagem_path"])

    t0 = time.time()
    response = analyze(img_path, patient, polish_laudo=True,
                        vision_model=vision_model, conduta_model=conduta_model)
    elapsed = time.time() - t0
    print(f"   pipeline: {response.status} | risco={response.risco} | conf={response.confianca_global:.0%} | {elapsed:.1f}s")

    response_dict = response.model_dump()
    response_dict["findings"] = response.findings.model_dump() if response.findings else None
    run_id = db.insert_run("eval", cid, response.pipeline_version, response_dict)

    if response.status != "ok" or response.findings is None:
        print(f"   [warn] sem findings ({response.erro}) — pulando comparação")
        return {"run_id": run_id, "status": response.status}

    # Parse GT e compara
    try:
        gt_findings = _ensure_gt_findings(cid, case_row["laudo_gt"], reparse=reparse_gt)
    except GTParseError as e:
        print(f"   [erro] GT parse falhou: {e}")
        return {"run_id": run_id, "status": "gt_parse_failed", "erro": str(e)}

    cmp_result = compare(response.findings, gt_findings, case_id=cid)
    cmp_dict = cmp_result.model_dump()
    counters = {
        "n_agree": cmp_result.n_agree,
        "n_catch": cmp_result.n_catch,
        "n_miss": cmp_result.n_miss,
        "n_hallucination": cmp_result.n_hallucination,
        "n_neutral": cmp_result.n_neutral_disagree,
    }
    cmp_id = db.insert_comparison(run_id, cid, cmp_dict, counters)

    # Persiste CATCH/HALLUCINATION para revisão
    for item in cmp_result.itens:
        if item.categoria in ("CATCH", "HALLUCINATION") and item.peso_clinico in ("medio", "alto"):
            db.insert_catch_review(
                comparison_id=cmp_id,
                case_id=cid,
                campo=item.campo,
                sistema=item.sistema,
                gt=item.gt,
            )

    print(f"   compare: AGREE={cmp_result.n_agree} CATCH={cmp_result.n_catch} "
          f"MISS={cmp_result.n_miss} HALLUC={cmp_result.n_hallucination} "
          f"NEUTRAL={cmp_result.n_neutral_disagree}")
    return {"run_id": run_id, "comparison_id": cmp_id, "status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rerodar mesmo casos já com run")
    parser.add_argument("--reparse-gt", action="store_true", help="re-extrair findings do GT")
    parser.add_argument("--only", type=str, default="", help="lista de IDs separados por vírgula")
    parser.add_argument("--vision-model", type=str, default=None, help="ex: claude-haiku-4-5-20251001")
    parser.add_argument("--conduta-model", type=str, default=None)
    args = parser.parse_args()

    db.init_db()
    cases = db.list_eval_cases()
    if not cases:
        print("Nenhum eval_case importado. Rode: python -m bootstrap.import_xlsx")
        return

    if args.only:
        ids = {int(x) for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c["id"] in ids]

    print(f"Rodando pipeline em {len(cases)} casos...")
    results = {"ok": 0, "skipped": 0, "erro": 0}
    for case in cases:
        try:
            r = run_one(case, reparse_gt=args.reparse_gt, force_run=args.force,
                        vision_model=args.vision_model, conduta_model=args.conduta_model)
            if r.get("skipped"):
                results["skipped"] += 1
            elif r.get("status") == "ok":
                results["ok"] += 1
            else:
                results["erro"] += 1
        except Exception as e:
            print(f"   [crash] {e}")
            results["erro"] += 1

    print("\n" + "=" * 50)
    print(f"OK: {results['ok']} | Skipped: {results['skipped']} | Erro: {results['erro']}")
    print(f"Para o relatório: python -m eval.report")


if __name__ == "__main__":
    main()
