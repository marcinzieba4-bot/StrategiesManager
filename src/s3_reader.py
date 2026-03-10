"""S3 file reading utilities for the Telegram Lambda agent."""
from __future__ import annotations

import logging
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Reader:
    """Reads files and lists keys from an S3 bucket."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        default_prefix: str = "",
    ) -> None:
        self.bucket = bucket
        self.default_prefix = default_prefix
        self._client = boto3.client("s3", region_name=region)

    # ── Public API ──────────────────────────────────────────────────────────

    def list_files(self, prefix: Optional[str] = None, max_keys: int = 100) -> list[str]:
        """Return a list of object keys under *prefix* (defaults to default_prefix)."""
        effective_prefix = prefix if prefix is not None else self.default_prefix
        keys: list[str] = []

        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(
                Bucket=self.bucket,
                Prefix=effective_prefix,
                PaginationConfig={"MaxItems": max_keys, "PageSize": min(max_keys, 1000)},
            ):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except ClientError as exc:
            logger.error("S3 list error (bucket=%s prefix=%s): %s", self.bucket, effective_prefix, exc)
            raise

        logger.info("Listed %d keys under s3://%s/%s", len(keys), self.bucket, effective_prefix)
        return keys

    def read_file(self, key: str, max_bytes: int = 80_000) -> str:
        """Download and return the text content of *key* (up to *max_bytes*)."""
        logger.info("Reading s3://%s/%s", self.bucket, key)
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"s3://{self.bucket}/{key} does not exist") from exc
            raise

        body = response["Body"].read(max_bytes)
        content_type = response.get("ContentType", "")

        # Attempt UTF-8, fall back to latin-1 for binary-ish files
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            text = body.decode("latin-1")

        truncated = response["ContentLength"] > max_bytes
        if truncated:
            text += f"\n\n[… file truncated at {max_bytes:,} bytes …]"

        logger.info(
            "Read %d bytes from s3://%s/%s (content-type: %s, truncated: %s)",
            len(body),
            self.bucket,
            key,
            content_type,
            truncated,
        )
        return text

    def read_files_as_context(self, prefix: Optional[str] = None, max_files: int = 10) -> str:
        """Concatenate the content of multiple S3 files into a single context string."""
        keys = self.list_files(prefix=prefix, max_keys=max_files)
        if not keys:
            return "No files found."

        parts: list[str] = []
        for key in keys[:max_files]:
            try:
                content = self.read_file(key)
                parts.append(f"### File: {key}\n\n{content}")
            except Exception as exc:  # noqa: BLE001
                parts.append(f"### File: {key}\n\n[Error reading file: {exc}]")

        return "\n\n---\n\n".join(parts)
