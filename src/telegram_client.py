"""Thin wrapper around the Telegram Bot HTTP API."""
from __future__ import annotations

import logging
from typing import Optional

import urllib3

logger = logging.getLogger(__name__)

_http = urllib3.PoolManager()
_BASE = "https://api.telegram.org/bot{token}/{method}"

# Telegram message length limit
_MAX_MSG_LEN = 4096


class TelegramClient:
    """Sends messages and actions to a Telegram chat."""

    def __init__(self, token: str) -> None:
        self._token = token

    # ── Public API ──────────────────────────────────────────────────────────

    def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> dict:
        """Send a text message, splitting it if it exceeds Telegram's 4096-char limit."""
        chunks = _split_message(text)
        last_response: dict = {}
        for chunk in chunks:
            payload: dict = {"chat_id": chat_id, "text": chunk}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            last_response = self._call("sendMessage", payload)
        return last_response

    def send_chat_action(self, chat_id: int, action: str = "typing") -> dict:
        """Show a typing / upload indicator in the chat."""
        return self._call("sendChatAction", {"chat_id": chat_id, "action": action})

    def set_webhook(
        self,
        url: str,
        secret_token: Optional[str] = None,
        allowed_updates: Optional[list[str]] = None,
    ) -> dict:
        """Register the webhook URL with Telegram."""
        payload: dict = {"url": url}
        if secret_token:
            payload["secret_token"] = secret_token
        if allowed_updates:
            payload["allowed_updates"] = allowed_updates
        return self._call("setWebhook", payload)

    def delete_webhook(self) -> dict:
        return self._call("deleteWebhook", {})

    def get_webhook_info(self) -> dict:
        return self._call("getWebhookInfo", {})

    def get_me(self) -> dict:
        return self._call("getMe", {})

    # ── Internal ────────────────────────────────────────────────────────────

    def _call(self, method: str, payload: dict) -> dict:
        import json

        url = _BASE.format(token=self._token, method=method)
        encoded = json.dumps(payload).encode()
        response = _http.request(
            "POST",
            url,
            body=encoded,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        data = json.loads(response.data.decode())

        if not data.get("ok"):
            logger.error("Telegram %s failed: %s", method, data)
        else:
            logger.debug("Telegram %s succeeded", method)

        return data


# ── Helpers ─────────────────────────────────────────────────────────────────

def _split_message(text: str, limit: int = _MAX_MSG_LEN) -> list[str]:
    """Split *text* into chunks that fit within Telegram's character limit."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at the last newline within the limit
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks
