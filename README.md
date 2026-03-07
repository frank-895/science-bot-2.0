# science-bot
An AI-powered agent that analyses life science datasets and answers scientific questions.

## How to use science-bot

1. Clone the repository:

```bash
git clone https://github.com/frank-895/science-bot.git
```

2. Configure `.env` from `.env.example` with `OPENAI_API_KEY`.
3. Start executor workers:

```bash
docker compose up -d --scale runner=4
```

## Run benchmark

```bash
uv --project science-bot run science-bot benchmark \
  --directory <path/to/capsule_folders.zip> \
  --csv <path/to/BixBenchFiltered_50_clean.csv>
```

Tracing is written to:

```bash
.science-bot/traces
```

The trace directory is cleaned at the start of each benchmark run.

Stop workers:

```bash
docker compose down
```
