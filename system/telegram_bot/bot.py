"""Bot Telegram ECG.

Fluxo:
  1. Foto + legenda (dados paciente) → GRUPO_ECG_ID
  2. Bot armazena imagem + production_case + roda pipeline
  3. Bot posta laudo+conduta+risco no GRUPO_REVISAO_ID (com foto + botões textuais 1/2)
  4. Médica responde "1" (aprovar) ou "2" (corrigir)
  5. Se "2", médica envia próxima mensagem com versão corrigida
  6. Decisão salva → publica laudo final no GRUPO_ECG_ID

TUDO armazenado: image, patient text, response, decisão. Production cases ficam
em tabela separada — disponíveis para retraining/fewshot, sem leak com eval_cases.

Uso: python -m telegram_bot.bot"""

import asyncio
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from config import (
    GRUPO_ECG_ID, GRUPO_REVISAO_ID, PROD_IMAGES_DIR, TELEGRAM_BOT_TOKEN,
)
from schemas import PatientData
from pipeline.orchestrator import analyze
from storage import db
from telegram_bot.auth import is_allowed
from telegram_bot.tg_api import (
    download_file, get_updates, send_photo, send_text,
)

DIVISOR = "━━━━━━━━━━━━━━━"

# Estado é persistido no SQLite (tabela pending_reviews). Não há dict em memória.
# Quando o processo reinicia, o bot recupera os reviews pendentes do disco.

# Caracteres especiais do Markdown V1 do Telegram. Se aparecerem desbalanceados
# (ex: um único `_` em "queixa_critica_presente"), o sendMessage falha com 400
# "can't parse entities". Texto dinâmico (caso do paciente, laudo, conduta,
# nomes de regras/padrões) precisa ser escapado — texto do template não, pois
# nele os `*` e `` ` `` são intencionais (negrito / monoespaçado).
_MD_ESCAPE_RE = re.compile(r"([_*\[\]`])")


def _md_safe(s: str) -> str:
    return _MD_ESCAPE_RE.sub(r"\\\1", s or "")


def _gen_external_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def _now_str() -> str:
    return datetime.now().strftime("%d/%m/%Y às %H:%M")


def _save_image(external_id: str, image_bytes: bytes) -> str:
    p = PROD_IMAGES_DIR / f"{external_id}.jpeg"
    p.write_bytes(image_bytes)
    return str(p)


def _fmt_padrao(p: str) -> str:
    """ecg_normal/stemi_inferior/bre → 'normal', 'stemi inferior', 'BRE'."""
    if p == "ecg_normal":
        return "normal"
    return p.replace("_", " ")


def _msg_revisao(external_id: str, patient_text: str, laudo: str, conduta: str,
                 risco: str, conf: int, alertas: list[str],
                 padroes: list[str] | None = None,
                 regras: list[str] | None = None) -> str:
    risco_label = "INTERMEDIÁRIO" if risco == "INTERMEDIARIO" else (
        "EMERGÊNCIA" if risco == "EMERGENCIA" else risco
    )
    parts = [
        f"🔬 *Novo ECG para revisão*",
        f"🆔 `{external_id}`",
        f"👤 {_md_safe(patient_text)}",
        f"🕐 {_now_str()}",
        DIVISOR,
        f"📄 *LAUDO*",
        _md_safe(laudo),
        DIVISOR,
        f"💊 *CONDUTA*",
        _md_safe(conduta),
        DIVISOR,
        f"*RISCO:* {risco_label}  |  *Conf:* {conf}%",
    ]
    if alertas:
        parts.append(f"⚠️ {_md_safe('; '.join(alertas))}")
    # Justificativa: mostra os padrões identificados e regras de risco que
    # dispararam. Omite no caso trivial (só ecg_normal sem regra) pra não
    # poluir a revisão de ECG limpo. Médica usa pra decidir se foi clínica
    # (erro de raciocínio) ou só estilo ao escolher 2/3/4.
    padroes_str = [p for p in (padroes or []) if p and p != "ecg_normal"]
    regras_str = [r for r in (regras or []) if r]
    if padroes_str or regras_str:
        parts.append(DIVISOR)
        parts.append("🧠 *Raciocínio do sistema*")
        if padroes_str:
            parts.append(f"padrões: {_md_safe(', '.join(_fmt_padrao(p) for p in padroes_str[:6]))}")
        if regras_str:
            parts.append(f"regras de risco: {_md_safe('; '.join(regras_str[:4]))}")
    parts.append(DIVISOR)
    parts.append(
        "Responda:\n"
        "*1* — aprovar\n"
        "*2* — corrigir só estilo (laudo/conduta clinicamente certos, palavras diferentes)\n"
        "*3* — corrigir clínica (achado ou conduta diferente do que escrevi)\n"
        "*4* — corrigir grave (a versão acima teria prejudicado o paciente)"
    )
    return "\n".join(parts)


# Mapa dos botões textuais 2/3/4 para o tipo de correção. Usado no bot e no
# fewshot/eval (correções 'grave' contam como sinal de incidente).
CORRECTION_TYPE_BY_BUTTON = {"2": "estilo", "3": "clinico", "4": "grave"}


def _msg_aprovado(external_id: str, patient_text: str, laudo: str, conduta: str,
                  laudo_alterado: bool = True, conduta_alterada: bool = True) -> str:
    """Render do laudo final. Quando vem de aprovação direta, ambos os flags
    são True (não muda nada na UI). Quando vem de correção parcial, o lado
    NÃO alterado recebe um marcador "(mantido do sistema)" pra a médica saber
    o que ela aprovou implicitamente."""
    laudo_tag = "" if laudo_alterado else "  _(mantido do sistema)_"
    conduta_tag = "" if conduta_alterada else "  _(mantida do sistema)_"
    return "\n".join([
        f"✅ *Laudo validado*",
        f"🆔 `{external_id}`",
        f"👤 {_md_safe(patient_text)}",
        f"🕐 {_now_str()}",
        DIVISOR,
        f"📄 *LAUDO*{laudo_tag}",
        _md_safe(laudo),
        DIVISOR,
        f"💊 *CONDUTA*{conduta_tag}",
        _md_safe(conduta),
        DIVISOR,
        f"⚠️ Uso médico. Validação clínica necessária.",
    ])


# Rótulo LAUDO/CONDUTA no início de uma linha, tolerando prefixo não-textual
# (emoji 📄/💊, markdown *, espaços) e separador à direita (:, *, espaços, quebra).
_LABEL_RE = re.compile(r"(?im)^[^\w\n]*(LAUDO|CONDUTA)\b[:*\s]*")


def _trim_segmento(seg: str) -> str:
    """Remove linhas das bordas que só têm pontuação/divisor (━, *, —) — sem letras."""
    linhas = seg.strip().splitlines()
    while linhas and not re.search(r"\w", linhas[0]):
        linhas.pop(0)
    while linhas and not re.search(r"\w", linhas[-1]):
        linhas.pop()
    return "\n".join(linhas).strip()


def _separa_laudo_conduta(texto: str) -> tuple[str, str]:
    """Separa o texto da médica em (laudo, conduta) pelos rótulos LAUDO/CONDUTA.

    Tolera prefixos (emoji, markdown, espaços) antes do rótulo. Cada seção é o
    texto até o próximo rótulo. Sem rótulo nenhum → ('' , '') (caller decide).
    Cada seção pode vir vazia se o rótulo correspondente não aparecer."""
    matches = list(_LABEL_RE.finditer(texto))
    if not matches:
        return "", ""
    laudo, conduta = "", ""
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(texto)
        seg = _trim_segmento(texto[m.end():end])
        if m.group(1).upper() == "LAUDO":
            laudo = seg
        else:
            conduta = seg
    return laudo, conduta


# ── handlers ──────────────────────────────────────────────────────────────────


async def _run_pipeline(
    image_bytes: bytes,
    image_path: str,
    caption: str,
    chat_id: int,
    user_id: Optional[int],
) -> None:
    """Persiste o caso, roda o pipeline e publica no grupo de revisão."""
    external_id = _gen_external_id()
    logger.info("Novo ECG {} de chat {}", external_id, chat_id)

    production_case_id = db.insert_production_case(
        external_id=external_id,
        caso_texto=caption,
        imagem_path=image_path,
        chat_id=chat_id,
        medico_user_id=user_id,
    )

    patient = PatientData(queixa=caption, contexto=caption)
    try:
        response = await asyncio.to_thread(analyze, image_bytes, patient, True)
    except Exception as e:
        logger.exception("pipeline crash {}", external_id)
        await send_text(GRUPO_REVISAO_ID,
                        f"⚠️ Pipeline falhou para `{external_id}`: {e}")
        return

    response_dict = response.model_dump()
    response_dict["findings"] = response.findings.model_dump() if response.findings else None
    run_id = db.insert_run("production", production_case_id,
                            response.pipeline_version, response_dict)

    if response.status != "ok":
        msg = f"❌ `{external_id}`\n{response.whatsapp()}"
        await send_text(GRUPO_REVISAO_ID, msg)
        return

    conf = int(response.confianca_global * 100)
    review_text = _msg_revisao(
        external_id, caption,
        response.laudo, response.conduta,
        response.risco, conf,
        response.justificativa.alertas_vitais,
        padroes=response.justificativa.padroes_identificados,
        regras=response.justificativa.regras_disparadas,
    )
    # Foto com legenda curta + laudo completo como texto-reply (captions têm limite
    # de 1024 chars; texto vai até 4096, então cabe paciente longo + laudo + conduta).
    photo_msg_id = await send_photo(
        GRUPO_REVISAO_ID, image_bytes,
        caption=f"🔬 ECG `{external_id}` — laudo abaixo",
    )
    review_msg_id = await send_text(
        GRUPO_REVISAO_ID, review_text, reply_to=photo_msg_id,
    )

    db.upsert_pending_review(
        external_id=external_id,
        production_case_id=production_case_id,
        run_id=run_id,
        review_msg_id=review_msg_id,
        step="aguardando_decisao",
        laudo=response.laudo,
        conduta=response.conduta,
        risco=response.risco,
        patient_text=caption,
        image_path=image_path,
    )


async def handle_photo(message: dict) -> None:
    chat_id = message["chat"]["id"]
    user_id = message.get("from", {}).get("id")
    caption = (message.get("caption") or "").strip()

    # Maior resolução
    photo = max(message["photo"], key=lambda p: p.get("file_size", 0))
    image_bytes = await download_file(photo["file_id"])
    if not image_bytes:
        await send_text(GRUPO_REVISAO_ID, "⚠️ Falha ao baixar imagem.")
        return

    # Caso 1: foto já vem com legenda → processa direto
    if caption:
        img_path = _save_image(_gen_external_id(), image_bytes)
        await _run_pipeline(image_bytes, img_path, caption, chat_id, user_id)
        return

    # Caso 2: foto sem legenda + texto já pendente → combina e processa
    pending = db.get_pending_input(chat_id, user_id) if user_id is not None else None
    if pending and pending["caption_text"]:
        text = pending["caption_text"]
        db.delete_pending_input(chat_id, user_id)
        img_path = _save_image(_gen_external_id(), image_bytes)
        await _run_pipeline(image_bytes, img_path, text, chat_id, user_id)
        return

    # Caso 3: foto sem legenda e sem texto pendente → guarda foto e pede dados
    if user_id is None:
        await send_text(chat_id, "⚠️ Não consegui identificar o remetente.")
        return
    img_path = _save_image(_gen_external_id(), image_bytes)
    db.upsert_pending_input(chat_id, user_id, image_path=img_path, caption_text=None)
    await send_text(chat_id,
                    "📸 Foto recebida. Envie agora, em uma mensagem, os dados do paciente "
                    "(idade, sexo, queixa, sinais vitais, comorbidades).")


async def handle_text_ecg(message: dict) -> None:
    """Texto no grupo ECG: dados do paciente, possivelmente antes da foto."""
    chat_id = message["chat"]["id"]
    user_id = message.get("from", {}).get("id")
    text = (message.get("text") or "").strip()

    if not text or user_id is None:
        return

    pending = db.get_pending_input(chat_id, user_id)

    # Caso 1: texto + foto pendente → combina e processa
    if pending and pending["image_path"]:
        img_path = pending["image_path"]
        try:
            image_bytes = Path(img_path).read_bytes()
        except FileNotFoundError:
            db.delete_pending_input(chat_id, user_id)
            await send_text(chat_id, "⚠️ Imagem pendente não encontrada. Reenvie a foto.")
            return
        db.delete_pending_input(chat_id, user_id)
        await _run_pipeline(image_bytes, img_path, text, chat_id, user_id)
        return

    # Caso 2: texto sem foto pendente → guarda texto e pede foto
    db.upsert_pending_input(chat_id, user_id, image_path=None, caption_text=text)
    await send_text(chat_id, "📝 Dados recebidos. Envie agora a foto do ECG.")


async def handle_text_review(message: dict) -> None:
    texto = (message.get("text") or "").strip()
    user_id = message.get("from", {}).get("id")

    # Identifica o ECG: prioriza reply; senão usa o único pendente desta médica.
    reply = message.get("reply_to_message") or {}
    reply_msg_id = reply.get("message_id")

    pending = None
    if reply_msg_id:
        pending = db.get_pending_by_msg(reply_msg_id)
        # Ownership: só a médica dona do caso pode aprovar/corrigir o próprio.
        # Se outra responder por reply (ou alguém sem user_id, ex.: admin anônimo),
        # rejeita com mensagem clara e não processa.
        if pending and pending["medico_user_id"] is not None \
                and pending["medico_user_id"] != user_id:
            logger.info("revisao: user {} tentou responder caso {} de owner {} — bloqueado",
                        user_id, pending["external_id"], pending["medico_user_id"])
            await send_text(
                GRUPO_REVISAO_ID,
                f"`{pending['external_id']}`: esse caso é de outra médica. "
                f"Apenas quem enviou a foto pode aprovar ou corrigir.",
            )
            return

    if not pending and user_id is not None:
        meus_pendentes = db.list_pending_for_user(user_id)
        if len(meus_pendentes) == 1:
            pending = meus_pendentes[0]
        elif len(meus_pendentes) > 1:
            logger.info("revisao: user {} resp={!r} sem reply, mas tem {} pendentes",
                        user_id, texto[:20], len(meus_pendentes))
            if texto in ("1", "2", "3", "4"):
                await send_text(
                    GRUPO_REVISAO_ID,
                    f"Você tem {len(meus_pendentes)} casos aguardando decisão. "
                    f"Responda *à mensagem* do caso específico com 1/2/3/4.",
                )
            return

    if not pending:
        # Sem reply e sem pendente desta médica: ignora silenciosamente (pode
        # ser conversa interna do grupo). Log só pra debug se vier 1/2/3/4 —
        # nesse caso é provável que ela achou que estava respondendo a um caso
        # mas o pending já foi resolvido ou está fora do escopo dela.
        if user_id and texto in ("1", "2", "3", "4"):
            logger.info("revisao: user {} resp={!r} sem reply e sem pendente — ignorado",
                        user_id, texto)
        return

    logger.info("revisao: user {} resp={!r} -> caso {} step={}",
                user_id, texto[:20], pending["external_id"], pending["step"])

    external_id = pending["external_id"]
    step = pending["step"]

    if step == "aguardando_decisao":
        if texto == "1":
            await _aprovar(pending, user_id)
        elif texto in CORRECTION_TYPE_BY_BUTTON:
            ctype = CORRECTION_TYPE_BY_BUTTON[texto]
            db.set_pending_step(external_id, "aguardando_edicao", correction_type=ctype)
            rotulo = {"estilo": "ESTILO", "clinico": "CLÍNICA",
                      "grave": "CLÍNICA GRAVE"}[ctype]
            await send_text(GRUPO_REVISAO_ID,
                            f"✏️ `{external_id}` — correção de *{rotulo}*\n"
                            f"Envie em uma única mensagem. Pode corrigir SÓ uma das duas:\n"
                            f"*LAUDO*\n[texto] — se for corrigir o laudo\n"
                            f"*CONDUTA*\n[texto] — se for corrigir a conduta\n"
                            f"O que não vier marcado mantém o que o sistema gerou.")
        else:
            await send_text(GRUPO_REVISAO_ID,
                            f"`{external_id}`: responda *1* (aprovar), "
                            f"*2* (estilo), *3* (clínica) ou *4* (grave).")
    elif step == "aguardando_edicao":
        await _corrigir(pending, texto, user_id)


def _load_image(image_path: str) -> bytes:
    return Path(image_path).read_bytes()


async def _aprovar(pending, user_id: Optional[int]) -> None:
    external_id = pending["external_id"]
    db.insert_decision(
        production_case_id=pending["production_case_id"],
        run_id=pending["run_id"],
        decision="aprovado",
        laudo_final=pending["laudo"],
        conduta_final=pending["conduta"],
        correcao_texto=None,
        medico_user_id=user_id,
    )
    msg = _msg_aprovado(external_id, pending["patient_text"], pending["laudo"], pending["conduta"])
    await _publicar_resultado(external_id, pending["image_path"], msg, prefixo="✅")
    db.delete_pending(external_id)
    logger.info("{} aprovado", external_id)


async def _corrigir(pending, texto: str, user_id: Optional[int]) -> None:
    external_id = pending["external_id"]
    laudo_corr, conduta_corr = _separa_laudo_conduta(texto)

    # Sem rótulo nenhum: não dá pra adivinhar o que ela quer corrigir, e reusar
    # a resposta inteira do modelo foi o erro do caso 5. Pede explícito.
    if not laudo_corr and not conduta_corr:
        await send_text(
            GRUPO_REVISAO_ID,
            f"`{external_id}`: não identifiquei rótulos *LAUDO* nem *CONDUTA* na "
            f"mensagem. Reenvie marcando o que está corrigindo (pode ser só um):\n"
            f"*LAUDO:*\n[texto]\nE/OU\n*CONDUTA:*\n[texto]",
        )
        return  # step continua 'aguardando_edicao' — próxima mensagem tenta de novo

    # Correção parcial: ela pode mandar SÓ um dos rótulos. O lado sem rótulo
    # mantém o que veio do modelo (gravado em pending). A confirmação publicada
    # logo abaixo deixa claro o que foi alterado e o que foi mantido.
    laudo_final = laudo_corr or pending["laudo"]
    conduta_final = conduta_corr or pending["conduta"]
    laudo_alterado = bool(laudo_corr)
    conduta_alterada = bool(conduta_corr)

    # correction_type vem do pending — botão que ela apertou (2/3/4). sqlite3.Row
    # não tem .get(); usa indexing direto (campo existe desde a migração).
    ctype = pending["correction_type"]
    db.insert_decision(
        production_case_id=pending["production_case_id"],
        run_id=pending["run_id"],
        decision="corrigido",
        laudo_final=laudo_final,
        conduta_final=conduta_final,
        correcao_texto=texto,
        medico_user_id=user_id,
        correction_type=ctype,
    )
    msg = _msg_aprovado(external_id, pending["patient_text"], laudo_final, conduta_final,
                        laudo_alterado=laudo_alterado, conduta_alterada=conduta_alterada)
    await _publicar_resultado(external_id, pending["image_path"], msg, prefixo="✅")
    db.delete_pending(external_id)
    logger.info("{} corrigido ({}) — laudo_alt={} conduta_alt={}",
                external_id, ctype or "tipo nao informado",
                laudo_alterado, conduta_alterada)


async def _publicar_resultado(external_id: str, image_path: str, msg: str, prefixo: str) -> None:
    """Publica foto + laudo no grupo ECG. Foto vai com legenda curta, laudo
    completo vai como texto-reply para caber casos longos (caption < 1024 chars,
    texto < 4096 chars)."""
    try:
        image_bytes = _load_image(image_path)
        photo_msg_id = await send_photo(
            GRUPO_ECG_ID, image_bytes,
            caption=f"{prefixo} `{external_id}` — laudo abaixo",
        )
        if photo_msg_id:
            await send_text(GRUPO_ECG_ID, msg, reply_to=photo_msg_id)
        else:
            await send_text(GRUPO_ECG_ID, msg)
    except Exception as e:
        logger.error("falha ao publicar {}: {}", external_id, e)
        await send_text(GRUPO_ECG_ID, msg)


# ── routing & polling ─────────────────────────────────────────────────────────


async def process_update(update: dict) -> None:
    message = update.get("message")
    if not message:
        return
    chat_id = message["chat"]["id"]
    if not is_allowed(chat_id):
        logger.warning("chat não autorizado: {}", chat_id)
        return

    if chat_id == GRUPO_ECG_ID and message.get("photo"):
        await handle_photo(message)
    elif chat_id == GRUPO_ECG_ID and message.get("text"):
        await handle_text_ecg(message)
    elif chat_id == GRUPO_REVISAO_ID and message.get("text"):
        await handle_text_review(message)


async def _drop_pending() -> int:
    """Descarta tudo que foi enviado antes do bot subir. Confirma os updates
    pendentes até o último e retorna o offset inicial (último update_id + 1).
    Sem updates pendentes → 0."""
    updates = await get_updates(offset=-1, timeout=0)
    if not updates:
        return 0
    last_id = updates[-1]["update_id"]
    await get_updates(offset=last_id + 1, timeout=0)  # confirma/descarta tudo até last_id
    logger.info("drop_pending: {} update(s) antigo(s) descartado(s), até update_id={}",
                len(updates), last_id)
    return last_id + 1


async def polling_loop(drop_pending: bool = True) -> None:
    offset = await _drop_pending() if drop_pending else 0
    logger.info("Polling iniciado. offset inicial={}", offset)
    while True:
        try:
            updates = await get_updates(offset=offset, timeout=20)
            for update in updates:
                try:
                    await process_update(update)
                except Exception as e:
                    logger.exception("erro update {}: {}", update.get("update_id"), e)
                offset = update["update_id"] + 1
        except Exception as e:
            logger.error("erro polling: {}", e)
            await asyncio.sleep(5)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        sys.exit("TELEGRAM_BOT_TOKEN não configurado no .env")
    if not GRUPO_ECG_ID or not GRUPO_REVISAO_ID:
        sys.exit("GRUPO_ECG_ID e GRUPO_REVISAO_ID precisam estar no .env")
    db.init_db()
    logger.info("ECG bot iniciando. ECG={} REV={}", GRUPO_ECG_ID, GRUPO_REVISAO_ID)
    asyncio.run(polling_loop())


if __name__ == "__main__":
    main()
