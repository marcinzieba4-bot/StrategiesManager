"""Claude-powered agent with S3 and Lambda tools."""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from lambda_invoker import LambdaInvoker
from s3_reader import S3Reader

logger = logging.getLogger(__name__)

# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "name": "list_s3_files",
        "description": (
            "List the files (object keys) stored in the S3 knowledge base. "
            "Use this to discover what context documents are available before reading them. "
            "Returns a list of file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": (
                        "S3 key prefix / folder to list. "
                        "Leave empty to use the default context prefix."
                    ),
                },
                "max_keys": {
                    "type": "integer",
                    "description": "Maximum number of keys to return (default 50, max 200).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "read_s3_file",
        "description": (
            "Read the text content of a specific file from the S3 knowledge base. "
            "Use list_s3_files first to find the correct key."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Full S3 object key (path) of the file to read.",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "invoke_lambda",
        "description": (
            "Trigger another AWS Lambda function. "
            "Use this to run automated strategies, fetch live data, or kick off pipelines. "
            "Returns the JSON response from the invoked function."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the Lambda function to invoke.",
                },
                "payload": {
                    "type": "object",
                    "description": "JSON payload to pass as the Lambda event (optional).",
                },
                "async_invoke": {
                    "type": "boolean",
                    "description": (
                        "If true, invoke asynchronously (fire-and-forget). "
                        "Default false (synchronous, waits for result)."
                    ),
                },
            },
            "required": ["function_name"],
        },
    },
]

# ── Default system prompt ────────────────────────────────────────────────────

_DEFAULT_SYSTEM = """\
You are a smart financial strategies assistant running inside AWS Lambda.
You have access to:
  • A knowledge base of files stored in S3 (use list_s3_files / read_s3_file).
  • The ability to trigger other Lambda functions (use invoke_lambda).

Guidelines:
  - Answer concisely and clearly.
  - When context from S3 would improve your answer, proactively read the relevant files.
  - When the user asks you to run a strategy or automation, use invoke_lambda.
  - Format responses using Markdown so they render nicely in Telegram.
  - If you cannot find relevant information, say so honestly.
"""


# ── Agent class ──────────────────────────────────────────────────────────────

class ClaudeAgent:
    """Agentic Claude loop with S3 and Lambda tools."""

    def __init__(
        self,
        s3_reader: S3Reader,
        lambda_invoker: LambdaInvoker,
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        system_prompt: str = "",
        max_iterations: int = 10,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._s3 = s3_reader
        self._lambda = lambda_invoker
        self._model = model
        self._max_tokens = max_tokens
        self._system = system_prompt.strip() or _DEFAULT_SYSTEM
        self._max_iterations = max_iterations

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(self, user_message: str, username: str = "User") -> str:
        """Run the agentic loop and return the final text reply."""
        logger.info("User '%s' sent: %s", username, user_message[:200])

        messages: list[dict] = [
            {"role": "user", "content": user_message},
        ]

        for iteration in range(self._max_iterations):
            logger.debug("Agent iteration %d", iteration + 1)

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                thinking={"type": "adaptive"},
                system=self._system,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            logger.info(
                "Claude stop_reason=%s, blocks=%d",
                response.stop_reason,
                len(response.content),
            )

            # Append assistant turn (full content list, not just text)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return _extract_text(response.content)

            if response.stop_reason == "tool_use":
                tool_results = self._execute_tools(response.content)
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "max_tokens":
                logger.warning("Hit max_tokens limit")
                return _extract_text(response.content) + "\n\n_(response cut off — token limit reached)_"

            # Unexpected stop reason
            logger.warning("Unexpected stop_reason: %s", response.stop_reason)
            return _extract_text(response.content)

        logger.warning("Reached max_iterations (%d)", self._max_iterations)
        return _extract_text(messages[-2]["content"]) if len(messages) >= 2 else "_(max iterations reached)_"

    # ── Tool execution ───────────────────────────────────────────────────────

    def _execute_tools(self, content: list[Any]) -> list[dict]:
        """Execute all tool_use blocks and return a list of tool_result dicts."""
        results: list[dict] = []

        for block in content:
            if block.type != "tool_use":
                continue

            tool_name: str = block.name
            tool_input: dict = block.input
            tool_use_id: str = block.id

            logger.info("Executing tool: %s(%s)", tool_name, json.dumps(tool_input)[:200])

            try:
                output = self._run_tool(tool_name, tool_input)
                result_content = json.dumps(output, ensure_ascii=False, default=str)
                is_error = False
            except Exception as exc:  # noqa: BLE001
                logger.exception("Tool %s raised an error", tool_name)
                result_content = f"Error executing {tool_name}: {exc}"
                is_error = True

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                    "is_error": is_error,
                }
            )

        return results

    def _run_tool(self, name: str, inputs: dict) -> Any:
        if name == "list_s3_files":
            prefix = inputs.get("prefix") or None
            max_keys = min(int(inputs.get("max_keys", 50)), 200)
            keys = self._s3.list_files(prefix=prefix, max_keys=max_keys)
            return {"files": keys, "count": len(keys)}

        if name == "read_s3_file":
            key: str = inputs["key"]
            content = self._s3.read_file(key)
            return {"key": key, "content": content}

        if name == "invoke_lambda":
            function_name: str = inputs["function_name"]
            payload: dict = inputs.get("payload") or {}
            async_invoke: bool = bool(inputs.get("async_invoke", False))

            if async_invoke:
                return self._lambda.invoke_async(function_name, payload)
            return self._lambda.invoke(function_name, payload)

        raise ValueError(f"Unknown tool: {name}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(content: Any) -> str:
    """Pull all text blocks out of a Claude content list."""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    items = content if isinstance(content, list) else [content]
    for block in items:
        if hasattr(block, "type"):
            if block.type == "text":
                parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "\n".join(parts).strip() or "_(no text response)_"
