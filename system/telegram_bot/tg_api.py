"""Wrapper async sobre a API HTTP do Telegram Bot."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from loguru import logger

from config import TELEGRAM_BOT_TOKEN

_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _scrub(s) -> str:
    """Remove o token do Telegram de qualquer string que vá pro log.

    httpx coloca a URL completa nas mensagens de erro (`Client error '409' for
    url 'https://api.telegram.org/bot<TOKEN>/getUpdates...'`). Se logar `e`
    direto, o token vaza pro `bot.err.log`. Esse helper substitui o token por
    `[REDACTED]` antes do log. Idempotente — também funciona se algum outro
    pacote logar formato parecido."""
    text = str(s)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN in text:
        text = text.replace(TELEGRAM_BOT_TOKEN, "[REDACTED]")
    return text


async def send_text(
    chat_id: int,
    text: str,
    reply_to: Optional[int] = None,
) -> Optional[int]:
    payload: dict = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{_BASE}/sendMessage", json=payload, timeout=30)
            r.raise_for_status()
            return r.json()["result"]["message_id"]
    except httpx.HTTPStatusError as e:
        logger.error("send_text falhou ({}): {} body={}",
                     chat_id, _scrub(e), _scrub(e.response.text[:300]))
        return None
    except Exception as e:
        logger.error("send_text falhou ({}): {}", chat_id, _scrub(e))
        return None


async def send_photo(
    chat_id: int,
    image_bytes: bytes,
    caption: str = "",
    reply_to: Optional[int] = None,
) -> Optional[int]:
    data: dict = {"chat_id": str(chat_id), "caption": caption, "parse_mode": "Markdown"}
    if reply_to:
        data["reply_to_message_id"] = str(reply_to)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(
                f"{_BASE}/sendPhoto",
                data=data,
                files={"photo": ("ecg.jpg", image_bytes, "image/jpeg")},
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["result"]["message_id"]
    except httpx.HTTPStatusError as e:
        logger.error("send_photo falhou ({}): {} body={}",
                     chat_id, _scrub(e), _scrub(e.response.text[:300]))
        return None
    except Exception as e:
        logger.error("send_photo falhou ({}): {}", chat_id, _scrub(e))
        return None


async def download_file(file_id: str) -> Optional[bytes]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{_BASE}/getFile", params={"file_id": file_id}, timeout=10)
            r.raise_for_status()
            file_path = r.json()["result"]["file_path"]
            dl = await c.get(
                f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}",
                timeout=60,
            )
            dl.raise_for_status()
            return dl.content
    except Exception as e:
        logger.error("download_file falhou ({}): {}", file_id, _scrub(e))
        return None


async def get_updates(offset: int = 0, timeout: int = 20) -> list[dict]:
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(
                f"{_BASE}/getUpdates",
                params={"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
                timeout=timeout + 10,
            )
            r.raise_for_status()
            return r.json().get("result", [])
    except httpx.ReadTimeout:
        return []
    except Exception as e:
        # Falha de DNS/rede/HTTP retorna instantâneo — sem backoff vira hammer-loop.
        logger.error("get_updates falhou: {} — backoff 5s", _scrub(e))
        await asyncio.sleep(5)
        return []
