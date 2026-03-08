#!/usr/bin/env python3
"""
Register (or remove) the Telegram webhook for this bot.

Usage:
    # Set webhook
    python scripts/setup_webhook.py --url https://<api-gw-id>.execute-api.<region>.amazonaws.com/prod/webhook

    # Remove webhook
    python scripts/setup_webhook.py --delete

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET from .env (or environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

# Load .env if present
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # python-dotenv not installed; rely on real environment variables


def _api_call(token: str, method: str, payload: dict) -> dict:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def set_webhook(token: str, webhook_url: str, secret: str = "") -> None:
    payload: dict = {
        "url": webhook_url,
        "allowed_updates": ["message", "edited_message", "callback_query"],
        "drop_pending_updates": True,
    }
    if secret:
        payload["secret_token"] = secret

    print(f"Setting webhook → {webhook_url}")
    result = _api_call(token, "setWebhook", payload)
    _print_result(result)


def delete_webhook(token: str) -> None:
    print("Deleting webhook …")
    result = _api_call(token, "deleteWebhook", {"drop_pending_updates": True})
    _print_result(result)


def get_info(token: str) -> None:
    result = _api_call(token, "getWebhookInfo", {})
    _print_result(result)


def _print_result(result: dict) -> None:
    if result.get("ok"):
        print("✓ Success:", result.get("description") or result.get("result"))
    else:
        print("✗ Error:", result.get("description") or result)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Telegram webhook registration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", metavar="WEBHOOK_URL", help="Webhook URL to register")
    group.add_argument("--delete", action="store_true", help="Remove the current webhook")
    group.add_argument("--info", action="store_true", help="Print current webhook info")
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN is not set in environment or .env file.")
        sys.exit(1)

    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()

    if args.url:
        set_webhook(token, args.url, secret)
    elif args.delete:
        delete_webhook(token)
    elif args.info:
        get_info(token)

    # Always show current state
    print("\nCurrent webhook info:")
    get_info(token)


if __name__ == "__main__":
    main()
