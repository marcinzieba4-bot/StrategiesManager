"""
Telegram Lambda Agent — Main entry point.

Receives Telegram webhook POSTs via API Gateway, calls Claude with S3 context
and Lambda-invocation tools, and replies to the user in Telegram.

Environment variables (see .env.example):
    ANTHROPIC_API_KEY         Claude API key
    TELEGRAM_BOT_TOKEN        Telegram bot token from @BotFather
    TELEGRAM_WEBHOOK_SECRET   Optional shared secret for webhook verification
    S3_BUCKET_NAME            S3 bucket with context / knowledge files
    S3_CONTEXT_PREFIX         Key prefix for context files (default: "context/")
    CLAUDE_MODEL              Claude model ID (default: claude-opus-4-6)
    MAX_TOKENS                Max tokens per response (default: 4096)
    AGENT_SYSTEM_PROMPT       Optional custom system prompt
    AWS_REGION                AWS region (set automatically by Lambda runtime)
    LAMBDA_INVOKE_REGION      Region for Lambda-to-Lambda calls (default: AWS_REGION)
    LAMBDA_ALLOWED_FUNCTIONS  Comma-separated allow-list of callable Lambda names
    LOG_LEVEL                 Python log level (default: INFO)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

# Lazy imports: boto3 and anthropic are heavy; import at handler level so
# Lambda can report init errors cleanly.
from claude_agent import ClaudeAgent
from lambda_invoker import LambdaInvoker
from s3_reader import S3Reader
from telegram_client import TelegramClient

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(levelname)s %(name)s %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:  # noqa: ANN401
    """AWS Lambda entry point.

    Always returns HTTP 200 to Telegram (even on error) to prevent endless retries.
    """
    logger.debug("Raw event: %s", json.dumps(event, default=str)[:1000])

    # 1. Parse body
    try:
        body_str, update = _parse_event(event)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse event body: %s", exc)
        return _http(400, "Bad Request")

    # 2. Verify Telegram webhook secret
    if not _verify_secret(event, body_str):
        logger.warning("Webhook secret verification failed — dropping request")
        return _http(403, "Forbidden")

    # 3. Process the update (errors are caught so Telegram always gets 200)
    try:
        _process_update(update)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error while processing Telegram update")

    return _http(200, "OK")


# ── Update processing ─────────────────────────────────────────────────────────

def _process_update(update: dict) -> None:
    """Route a Telegram update to the appropriate handler."""
    message = update.get("message") or update.get("edited_message")
    if message:
        _handle_message(message)
        return

    if update.get("callback_query"):
        logger.info("Callback query received — not handled yet")
        return

    logger.info("Unhandled update type: %s", list(update.keys()))


def _handle_message(message: dict) -> None:
    """Process an incoming text message and reply via Claude."""
    chat_id: int = message["chat"]["id"]
    text: str = message.get("text", "").strip()
    user: dict = message.get("from", {})
    username: str = user.get("username") or user.get("first_name", "User")

    bot = TelegramClient(os.environ["TELEGRAM_BOT_TOKEN"])

    if not text:
        bot.send_message(chat_id, "Please send a text message.")
        return

    # Show typing indicator while we process
    bot.send_chat_action(chat_id, "typing")

    # Build agent
    region = os.environ.get("AWS_REGION", "us-east-1")
    agent = ClaudeAgent(
        s3_reader=S3Reader(
            bucket=os.environ["S3_BUCKET_NAME"],
            region=region,
            default_prefix=os.environ.get("S3_CONTEXT_PREFIX", "context/"),
        ),
        lambda_invoker=LambdaInvoker(
            region=os.environ.get("LAMBDA_INVOKE_REGION", region),
        ),
        model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-6"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "4096")),
        system_prompt=os.environ.get("AGENT_SYSTEM_PROMPT", ""),
    )

    # Run agent
    reply = agent.chat(user_message=text, username=username)

    # Send reply (auto-splits if > 4096 chars)
    bot.send_message(
        chat_id,
        reply,
        parse_mode="Markdown",
        reply_to_message_id=message.get("message_id"),
    )
    logger.info("Replied to chat_id=%s (%d chars)", chat_id, len(reply))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_event(event: dict) -> tuple[str, dict]:
    """Return (raw_body_str, parsed_update_dict)."""
    raw = event.get("body", event)
    if isinstance(raw, str):
        return raw, json.loads(raw)
    if isinstance(raw, dict):
        return json.dumps(raw), raw
    raise ValueError(f"Cannot parse body of type {type(raw)}")


def _verify_secret(event: dict, body_str: str) -> bool:
    """Verify the optional Telegram webhook secret token."""
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not secret:
        return True  # No verification configured

    headers: dict = event.get("headers") or {}
    # API Gateway may lowercase header names
    provided_token = (
        headers.get("X-Telegram-Bot-Api-Secret-Token")
        or headers.get("x-telegram-bot-api-secret-token")
        or ""
    )

    # Telegram signs with HMAC-SHA256 of the raw body using the secret as key
    mac = hmac.new(secret.encode(), body_str.encode(), hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), provided_token)


def _http(status: int, body: str) -> dict:
    return {"statusCode": status, "body": body}
