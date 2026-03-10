"""
Telegram Lambda Agent — Main entry point.

Operating modes
───────────────
POLLING (default, recommended):
  EventBridge triggers this Lambda every minute.
  Lambda calls Telegram getUpdates, processes messages, stores the last
  update_id in S3 so the next invocation starts where this one left off.
  No public webhook URL required.

WEBHOOK (optional, requires accessible HTTPS endpoint):
  Telegram POSTs to Lambda Function URL.
  Handler immediately returns 200 and fires an async self-invocation so
  Claude has unlimited time to respond.

Admin invocations (call directly via boto3 / AWS CLI):
  {"admin_action": "poll"}            — run one polling cycle now
  {"admin_action": "delete_webhook"}  — remove any registered webhook
  {"admin_action": "get_webhook_info"}

Environment variables (see .env.example):
    ANTHROPIC_API_KEY         Claude API key
    TELEGRAM_BOT_TOKEN        Telegram bot token from @BotFather
    TELEGRAM_WEBHOOK_SECRET   Optional shared secret (webhook mode only)
    LAMBDA_FUNCTION_URL       HTTPS URL for webhook registration (webhook mode only)
    S3_BUCKET_NAME            S3 bucket with context / knowledge files
    S3_CONTEXT_PREFIX         Key prefix for context files (default: "Strategies/")
    S3_STATE_KEY              S3 key for polling offset state (default: "telegram-agent-state.json")
    CLAUDE_MODEL              Claude model ID (default: claude-opus-4-6)
    MAX_TOKENS                Max tokens per response (default: 4096)
    AGENT_SYSTEM_PROMPT       Optional custom system prompt
    AWS_REGION                AWS region (set automatically by Lambda runtime)
    LAMBDA_INVOKE_REGION      Region for Lambda-to-Lambda calls (default: AWS_REGION)
    LAMBDA_ALLOWED_FUNCTIONS  Comma-separated allow-list of callable Lambda names
    TELEGRAM_ALLOWED_CHAT_IDS Comma-separated allow-list of chat/user IDs (empty = all)
    LOG_LEVEL                 Python log level (default: INFO)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from typing import Any

from claude_agent import ClaudeAgent
from lambda_invoker import LambdaInvoker
from s3_reader import S3Reader
from telegram_client import TelegramClient

# ── Logging ──────────────────────────────────────────────────────────────────

_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.getLogger().setLevel(_log_level)
logger = logging.getLogger(__name__)

# ── State keys ───────────────────────────────────────────────────────────────

_S3_STATE_KEY = os.environ.get("S3_STATE_KEY", "telegram-agent-state.json")


# ── Lambda handler ────────────────────────────────────────────────────────────

def lambda_handler(event: dict, context: Any) -> dict:  # noqa: ANN401
    """AWS Lambda entry point."""
    logger.debug("Raw event: %s", json.dumps(event, default=str)[:300])

    # ── Admin actions ────────────────────────────────────────────────────────
    if "admin_action" in event:
        return _handle_admin_action(event["admin_action"])

    # ── Async task (self-invoked in webhook mode) ────────────────────────────
    if "async_task" in event:
        return _handle_async_task(event["async_task"])

    # ── EventBridge polling trigger ──────────────────────────────────────────
    if event.get("source") == "aws.events" or event.get("detail-type") == "Scheduled Event":
        return _handle_poll()

    # ── Telegram webhook (Function URL POST) ─────────────────────────────────
    try:
        body_str, update = _parse_event(event)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse event body: %s", exc)
        return _http(400, "Bad Request")

    if not _verify_secret(event, body_str):
        logger.warning("Webhook secret verification failed — dropping request")
        return _http(403, "Forbidden")

    # Return 200 immediately; process async to beat Telegram's 30-second timeout
    try:
        _dispatch_async(update, context)
        logger.info("Dispatched update async, returning 200 immediately")
    except Exception:  # noqa: BLE001
        logger.warning("Async dispatch failed — processing synchronously")
        try:
            _process_update(update)
        except Exception:  # noqa: BLE001
            logger.exception("Unhandled error in synchronous fallback")

    return _http(200, "OK")


# ── Polling mode ──────────────────────────────────────────────────────────────

def _handle_poll() -> dict:
    """Fetch and process all new Telegram updates since the last run."""
    bot = TelegramClient(os.environ["TELEGRAM_BOT_TOKEN"])
    offset = _load_offset()

    logger.info("Polling Telegram updates (offset=%s)", offset)
    resp = bot.get_updates(offset=offset, limit=100)

    if not resp.get("ok"):
        logger.error("getUpdates failed: %s", resp)
        return _http(500, "getUpdates failed")

    updates = resp.get("result", [])
    logger.info("Received %d updates", len(updates))

    if not updates:
        return _http(200, "Processed 0 updates")

    max_update_id = max(u["update_id"] for u in updates)
    new_offset = max_update_id + 1

    # Acknowledge updates with Telegram BEFORE processing.
    # Calling getUpdates with the new offset tells Telegram these updates
    # are confirmed and must not be re-delivered. Saving to S3 only is not
    # enough — if S3 is stale or the invocation crashes, Telegram re-delivers.
    bot.get_updates(offset=new_offset, limit=1)
    _save_offset(new_offset)
    logger.info("Acknowledged updates up to update_id=%s (new_offset=%s)", max_update_id, new_offset)

    for update in updates:
        update_id = update["update_id"]
        try:
            _process_update(update)
        except Exception:  # noqa: BLE001
            logger.exception("Error processing update_id=%s", update_id)

    return _http(200, f"Processed {len(updates)} updates")


def _load_offset() -> int:
    """Load last processed update_id + 1 from S3."""
    import boto3
    bucket = os.environ["S3_BUCKET_NAME"]
    region = os.environ.get("AWS_REGION", "eu-north-1")
    try:
        s3 = boto3.client("s3", region_name=region)
        obj = s3.get_object(Bucket=bucket, Key=_S3_STATE_KEY)
        state = json.loads(obj["Body"].read())
        return int(state.get("next_offset", 0))
    except Exception:  # noqa: BLE001
        logger.info("No existing state in S3 — starting from offset 0")
        return 0


def _save_offset(next_offset: int) -> None:
    """Persist the next polling offset to S3."""
    import boto3
    bucket = os.environ["S3_BUCKET_NAME"]
    region = os.environ.get("AWS_REGION", "eu-north-1")
    try:
        s3 = boto3.client("s3", region_name=region)
        s3.put_object(
            Bucket=bucket,
            Key=_S3_STATE_KEY,
            Body=json.dumps({"next_offset": next_offset}).encode(),
            ContentType="application/json",
        )
        logger.info("Saved next_offset=%s to S3", next_offset)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to save offset to S3")


# ── Webhook async dispatch ────────────────────────────────────────────────────

def _dispatch_async(update: dict, context: Any) -> None:
    """Fire-and-forget: invoke this Lambda async so we return 200 to Telegram fast."""
    import boto3
    region = os.environ.get("AWS_REGION", "eu-north-1")
    client = boto3.client("lambda", region_name=region)
    client.invoke(
        FunctionName=context.invoked_function_arn,
        InvocationType="Event",
        Payload=json.dumps({"async_task": {"update": update}}).encode(),
    )


def _handle_async_task(task: dict) -> dict:
    update = task.get("update", {})
    try:
        _process_update(update)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in async task")
    return _http(200, "OK")


# ── Admin actions ─────────────────────────────────────────────────────────────

def _handle_admin_action(action: str) -> dict:
    bot = TelegramClient(os.environ["TELEGRAM_BOT_TOKEN"])

    if action == "get_webhook_info":
        info = bot.get_webhook_info()
        logger.info("Webhook info: %s", json.dumps(info))
        return _http(200, json.dumps(info))

    if action == "delete_webhook":
        result = bot.delete_webhook(drop_pending_updates=True)
        logger.info("Deleted webhook: %s", result)
        return _http(200, json.dumps(result))

    if action == "poll":
        return _handle_poll()

    if action == "create_schedule":
        return _create_eventbridge_schedule()

    if action == "setup_webhook":
        function_url = os.environ.get("LAMBDA_FUNCTION_URL", "").rstrip("/")
        if not function_url:
            return _http(500, "LAMBDA_FUNCTION_URL not set")
        result = bot.set_webhook(
            url=function_url,
            allowed_updates=["message", "edited_message", "callback_query"],
        )
        logger.info("set_webhook result: %s", result)
        return _http(200, json.dumps(result))

    logger.warning("Unknown admin_action: %s", action)
    return _http(400, f"Unknown action: {action}")


def _create_eventbridge_schedule() -> dict:
    """Create (or update) the EventBridge rule that polls Telegram every minute."""
    import boto3
    region = os.environ.get("AWS_REGION", "eu-north-1")
    function_arn = f"arn:aws:lambda:{region}:905418356298:function:telegram-agent"
    rule_name = "telegram-agent-poller"

    try:
        events = boto3.client("events", region_name=region)
        rule = events.put_rule(
            Name=rule_name,
            ScheduleExpression="rate(1 minute)",
            State="ENABLED",
            Description="Polls Telegram for new messages every minute",
        )
        rule_arn = rule["RuleArn"]
        events.put_targets(
            Rule=rule_name,
            Targets=[{"Id": "telegram-agent", "Arn": function_arn}],
        )
        # Grant EventBridge permission to invoke Lambda
        lamb = boto3.client("lambda", region_name=region)
        try:
            lamb.add_permission(
                FunctionName="telegram-agent",
                StatementId="allow-eventbridge-poller",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=rule_arn,
            )
        except lamb.exceptions.ResourceConflictException:
            pass  # Permission already exists
        logger.info("EventBridge rule created: %s", rule_arn)
        return _http(200, json.dumps({"ok": True, "rule_arn": rule_arn}))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create EventBridge rule")
        return _http(500, str(exc))


# ── Update processing ─────────────────────────────────────────────────────────

def _process_update(update: dict) -> None:
    message = update.get("message") or update.get("edited_message")
    if message:
        _handle_message(message)
        return
    if update.get("callback_query"):
        logger.info("Callback query — not handled")
        return
    logger.debug("Unhandled update type: %s", list(update.keys()))


def _is_allowed_chat(chat_id: int) -> bool:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").strip()
    if not raw:
        return True
    allowed = {s.strip() for s in raw.split(",") if s.strip()}
    return str(chat_id) in allowed


def _handle_message(message: dict) -> None:
    chat_id: int = message["chat"]["id"]
    text: str = message.get("text", "").strip()
    user: dict = message.get("from", {})
    username: str = user.get("username") or user.get("first_name", "User")

    bot = TelegramClient(os.environ["TELEGRAM_BOT_TOKEN"])

    if not _is_allowed_chat(chat_id):
        logger.warning("Blocked unauthorized chat_id=%s", chat_id)
        bot.send_message(chat_id, "Sorry, you are not authorized to use this bot.")
        return

    if not text:
        return

    logger.info("User '%s' (chat=%s) sent: %s", username, chat_id, text[:200])
    bot.send_chat_action(chat_id, "typing")

    region = os.environ.get("AWS_REGION", "eu-north-1")
    agent = ClaudeAgent(
        s3_reader=S3Reader(
            bucket=os.environ["S3_BUCKET_NAME"],
            region=region,
            default_prefix=os.environ.get("S3_CONTEXT_PREFIX", "Strategies/"),
        ),
        lambda_invoker=LambdaInvoker(
            region=os.environ.get("LAMBDA_INVOKE_REGION", region),
        ),
        model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-6"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "4096")),
        system_prompt=os.environ.get("AGENT_SYSTEM_PROMPT", ""),
    )

    reply = agent.chat(user_message=text, username=username)

    bot.send_message(
        chat_id,
        reply,
        parse_mode="Markdown",
        reply_to_message_id=message.get("message_id"),
    )
    logger.info("Replied to chat_id=%s (%d chars)", chat_id, len(reply))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_event(event: dict) -> tuple[str, dict]:
    raw = event.get("body", event)
    if isinstance(raw, str):
        return raw, json.loads(raw)
    if isinstance(raw, dict):
        return json.dumps(raw), raw
    raise ValueError(f"Cannot parse body of type {type(raw)}")


def _verify_secret(event: dict, body_str: str) -> bool:
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
    if not secret:
        return True
    headers: dict = event.get("headers") or {}
    provided = (
        headers.get("X-Telegram-Bot-Api-Secret-Token")
        or headers.get("x-telegram-bot-api-secret-token")
        or ""
    )
    mac = hmac.new(secret.encode(), body_str.encode(), hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), provided)


def _http(status: int, body: str) -> dict:
    return {"statusCode": status, "body": body}
