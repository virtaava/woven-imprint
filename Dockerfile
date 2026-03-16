FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir -e .

# Default: interactive demo (override with docker run args)
ENTRYPOINT ["woven-imprint"]
CMD ["demo"]
