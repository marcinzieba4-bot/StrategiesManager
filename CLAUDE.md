# StrategiesManager — Agent Rules

## S3 Knowledge Base Layout

```
s3://s3bucketmz/
  Strategies/
    json/   ← structured JSON reports (preferred for trading questions)
    *.pdf   ← original PDF reports (fallback)
```

## Rule: Always use JSON files for trading questions

When the user asks any trading or market question (outlook, sectors, macro,
strategy, positioning, signals, etc.), the agent **must**:

1. Call `list_s3_files` with prefix `Strategies/json/` first.
2. Read the **most recent** JSON file (highest date in filename).
3. Base the answer on that JSON content.
4. Only fall back to PDF files if no JSON is available.

**Reason:** JSON files are the machine-readable version of the daily Market
Intelligence briefing. They contain the same information as the PDFs but are
structured (sections → subheadings → content) and cheaper to read (no binary
decoding overhead).

## JSON file structure

```json
{
  "date": "Tuesday, March 10 2026",
  "generated_at": "2026-03-10T12:28:24Z",
  "sections": [
    {
      "heading": "Market Intelligence Briefing",
      "subheadings": ["🌍 Geographic Macro Overview", "📊 Sector Deep Dive", ...]
    },
    ...
  ]
}
```

## Deployment notes

- Runtime: AWS Lambda (arm64, Python 3.12), region `eu-north-1`
- Function name: `telegram-agent`
- Trigger: EventBridge rule `telegram-agent-poller` (every 1 minute, polling mode)
- Polling offset stored in: `s3://s3bucketmz/telegram-agent-state.json`
- IAM role: `CTAWeekly-role-9kzygc2c` (S3 GetObject/PutObject/ListBucket on s3bucketmz)
- Build command:
  ```bash
  pip install --platform manylinux2014_aarch64 --python-version 3.12 \
    --only-binary :all: --target .build/python/ -r src/requirements.txt
  cp src/*.py .build/python/
  cd .build/python && zip -r ../../lambda.zip .
  ```
