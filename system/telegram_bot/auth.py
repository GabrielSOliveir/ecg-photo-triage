"""Autorização: whitelist por chat_id (definida em .env como TELEGRAM_ALLOWED_CHAT_IDS).

Se a whitelist estiver vazia, apenas os grupos GRUPO_ECG_ID e GRUPO_REVISAO_ID
são aceitos (segurança mínima — sem aberta a todo mundo)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TELEGRAM_ALLOWED_CHAT_IDS, GRUPO_ECG_ID, GRUPO_REVISAO_ID


def is_allowed(chat_id: int) -> bool:
    if not chat_id:
        return False
    if chat_id == GRUPO_ECG_ID or chat_id == GRUPO_REVISAO_ID:
        return GRUPO_ECG_ID != 0 and GRUPO_REVISAO_ID != 0
    if TELEGRAM_ALLOWED_CHAT_IDS:
        return chat_id in TELEGRAM_ALLOWED_CHAT_IDS
    return False
