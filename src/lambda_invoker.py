"""Invoke other AWS Lambda functions from the Telegram agent."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class LambdaInvoker:
    """Wrapper around boto3 Lambda client for invoking sibling functions."""

    def __init__(self, region: str = "us-east-1") -> None:
        self._client = boto3.client("lambda", region_name=region)
        # Comma-separated allow-list; empty string = allow all
        raw_allow = os.environ.get("LAMBDA_ALLOWED_FUNCTIONS", "")
        self._allowed: set[str] = (
            {name.strip() for name in raw_allow.split(",") if name.strip()}
            if raw_allow.strip()
            else set()
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def invoke(
        self,
        function_name: str,
        payload: Optional[dict] = None,
        invocation_type: str = "RequestResponse",
    ) -> dict:
        """Invoke *function_name* and return its response payload.

        Args:
            function_name: Name or ARN of the target Lambda.
            payload: JSON-serialisable dict passed as the event.
            invocation_type: "RequestResponse" (sync) or "Event" (async / fire-and-forget).

        Returns:
            Parsed JSON response from the invoked Lambda, or an error dict.
        """
        if not self._is_allowed(function_name):
            msg = (
                f"Lambda function '{function_name}' is not in LAMBDA_ALLOWED_FUNCTIONS. "
                "Update the env var to permit this call."
            )
            logger.warning(msg)
            return {"error": msg}

        encoded = json.dumps(payload or {}).encode()
        logger.info("Invoking Lambda %s (type=%s)", function_name, invocation_type)

        try:
            response = self._client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=encoded,
            )
        except ClientError as exc:
            logger.error("Lambda invoke error for %s: %s", function_name, exc)
            return {"error": str(exc)}

        status_code: int = response.get("StatusCode", 0)

        if invocation_type == "Event":
            # Async: no payload returned
            return {"status": "accepted", "statusCode": status_code}

        raw_payload: bytes = response["Payload"].read()
        try:
            result = json.loads(raw_payload)
        except json.JSONDecodeError:
            result = {"raw": raw_payload.decode("utf-8", errors="replace")}

        if "FunctionError" in response:
            logger.error(
                "Lambda %s returned function error: %s (payload: %s)",
                function_name,
                response["FunctionError"],
                result,
            )
            result["_functionError"] = response["FunctionError"]

        logger.info("Lambda %s responded with status %d", function_name, status_code)
        return result

    def invoke_async(self, function_name: str, payload: Optional[dict] = None) -> dict:
        """Fire-and-forget invocation. Returns immediately without waiting for the result."""
        return self.invoke(function_name, payload, invocation_type="Event")

    def list_allowed_functions(self) -> list[str]:
        """Return the configured allow-list (empty = unrestricted)."""
        return sorted(self._allowed) if self._allowed else []

    # ── Internal ────────────────────────────────────────────────────────────

    def _is_allowed(self, function_name: str) -> bool:
        if not self._allowed:
            return True  # No restriction configured
        # Match on short name or full ARN suffix
        short = function_name.split(":")[-1]
        return function_name in self._allowed or short in self._allowed
