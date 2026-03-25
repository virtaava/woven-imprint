# Setup Guide — Docker

No Python installation needed. Docker handles everything.

## What You'll Need

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, Mac) or Docker Engine (Linux)
- About 5 GB of free disk space (container + AI model)

## Option 1: Docker Compose (recommended)

This starts both Woven Imprint and Ollama together — fully self-contained.

1. Clone the repo (or just download these two files: `Dockerfile` and `docker-compose.yml`):

```
git clone https://github.com/virtaava/woven-imprint.git
cd woven-imprint
```

2. Start everything:

```
docker compose up
```

3. Pull the AI models (first time only — run in a second terminal):

```
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull nomic-embed-text
```

4. Open your browser to **http://localhost:7860** — the demo UI is ready.

To stop: `Ctrl+C` or `docker compose down`

To restart later: `docker compose up` (your characters and models are saved in Docker volumes).

## Option 2: With Ollama on your host

If you already have Ollama installed and running on your machine:

**Linux** (host networking works):
```
docker build -t woven-imprint .
docker run --network host woven-imprint demo
```

**Mac / Windows** (use host.docker.internal):
```
docker build -t woven-imprint .
docker run -p 7860:7860 -e OLLAMA_HOST=http://host.docker.internal:11434 woven-imprint demo --port 7860
```

## Available Commands

```
docker compose exec woven-imprint woven-imprint list
docker compose exec woven-imprint woven-imprint create "Marcus"
docker compose exec woven-imprint woven-imprint chat marcus
docker compose exec woven-imprint woven-imprint stats marcus
```

Or if running standalone:
```
docker run --network host woven-imprint demo
docker run --network host woven-imprint serve --port 8650
docker run --network host woven-imprint --help
```

## Persistent Data

Docker Compose uses named volumes:
- `ollama_data` — downloaded AI models
- `woven_data` — your characters and memories

These survive `docker compose down`. To delete everything:
```
docker compose down -v
```

## Using OpenAI Instead of Ollama

If you don't want to run Ollama, set environment variables in your `docker-compose.yml`
or pass them to `docker run`:

```bash
docker run -p 7860:7860 \
  -e WOVEN_IMPRINT_LLM_PROVIDER=openai \
  -e WOVEN_IMPRINT_EMBEDDING_PROVIDER=openai \
  -e WOVEN_IMPRINT_API_KEY_LLM=sk-your-key-here \
  -e WOVEN_IMPRINT_MODEL=gpt-4o-mini \
  woven-imprint demo --port 7860
```

## What's Next?

- The demo UI at http://localhost:7860 has everything: chat, create, migrate, stats
- See [Getting Started](GETTING_STARTED.md) for more ways to use Woven Imprint
