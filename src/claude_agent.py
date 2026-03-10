"""Claude-powered agent with S3, Lambda, and web search tools."""
from __future__ import annotations

import html
import json
import logging
import os
import urllib.request
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus

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
        "name": "web_search",
        "description": (
            "Search the internet for up-to-date information not covered by the S3 knowledge base. "
            "Uses DuckDuckGo. Returns a summary and a list of relevant results with titles, snippets, and URLs. "
            "Use this ONLY when the S3 files do not contain the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch and return the plain-text content of a web page. "
            "Use this to read a specific article or page found via web_search."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the page to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default 8000).",
                },
            },
            "required": ["url"],
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
  • Internet search via DuckDuckGo (use web_search) and page fetching (use fetch_url).

## S3 Knowledge Base Layout
  Strategies/json/  — structured JSON market intelligence reports (PREFERRED)
  Strategies/       — original PDF reports (fallback only)

## RULE: For any trading or market question, always check JSON first
  1. Call list_s3_files with prefix "Strategies/json/" to find available reports.
  2. Read the most recent JSON file (highest date in filename, e.g. 2026-03-10_Market_Intelligence.json).
  3. Answer from that JSON content.
  4. Only fall back to PDFs if no JSON file exists.
  5. If neither S3 JSON nor PDF covers the topic, use web_search as a last resort.

The JSON files contain the same daily Market Intelligence briefing as the PDFs
but are fully structured (sections, subheadings, content) — much faster to read.

## General guidelines
  - Answer concisely and clearly.
  - When the user asks you to run a strategy or automation, use invoke_lambda.
  - Format responses using Markdown so they render nicely in Telegram.
  - If you cannot find relevant information in S3, search the internet before giving up.
  - Always indicate when your answer comes from an internet search rather than the internal knowledge base.
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

        if name == "web_search":
            return _web_search(inputs["query"])

        if name == "fetch_url":
            max_chars = int(inputs.get("max_chars", 8000))
            return _fetch_url(inputs["url"], max_chars)

        if name == "invoke_lambda":
            function_name: str = inputs["function_name"]
            payload: dict = inputs.get("payload") or {}
            async_invoke: bool = bool(inputs.get("async_invoke", False))

            if async_invoke:
                return self._lambda.invoke_async(function_name, payload)
            return self._lambda.invoke(function_name, payload)

        raise ValueError(f"Unknown tool: {name}")


# ── Web tools ────────────────────────────────────────────────────────────────

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TelegramAgent/1.0)",
    "Accept": "text/html,application/json",
}
_HTTP_TIMEOUT = 10


def _web_search(query: str) -> dict:
    """Search DuckDuckGo Instant Answer API (no key required)."""
    url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
    req = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: list[dict] = []

    # Abstract (direct answer)
    if data.get("Abstract"):
        results.append({
            "title": data.get("Heading", ""),
            "snippet": data["Abstract"],
            "url": data.get("AbstractURL", ""),
            "source": data.get("AbstractSource", ""),
        })

    # Related topics
    for topic in data.get("RelatedTopics", [])[:8]:
        if "Text" in topic:
            results.append({
                "title": topic.get("Text", "")[:120],
                "snippet": topic.get("Text", ""),
                "url": topic.get("FirstURL", ""),
            })
        # sub-topics (Topics list inside a topic)
        for sub in topic.get("Topics", [])[:3]:
            if "Text" in sub:
                results.append({
                    "title": sub.get("Text", "")[:120],
                    "snippet": sub.get("Text", ""),
                    "url": sub.get("FirstURL", ""),
                })

    return {
        "query": query,
        "answer": data.get("Answer") or data.get("Abstract") or "",
        "results": results[:10],
    }


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain text converter."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "nav", "footer", "head"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "head"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return html.unescape("".join(self._parts))


def _fetch_url(url: str, max_chars: int = 8000) -> dict:
    """Fetch a URL and return its plain-text content."""
    req = urllib.request.Request(url, headers=_REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read(max_chars * 4)  # over-read then trim after decoding

    text: str
    if "html" in content_type:
        parser = _TextExtractor()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.get_text()
    else:
        text = raw.decode("utf-8", errors="replace")

    # Collapse whitespace and trim
    import re
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    truncated = len(text) > max_chars
    return {
        "url": url,
        "content": text[:max_chars],
        "truncated": truncated,
    }


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
