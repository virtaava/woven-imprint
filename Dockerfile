FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Default: show help. Use with --network host and Ollama running on the host:
#   docker run --network host woven-imprint demo
#   docker run --network host woven-imprint serve --port 8650
ENTRYPOINT ["woven-imprint"]
CMD ["--help"]
