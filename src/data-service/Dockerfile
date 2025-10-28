# --- Build Stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install poetry

COPY ../../pyproject.toml ../../poetry.lock* ./

RUN poetry install --with fetch-data

COPY ./fetch_data.py ./utils.py ./

# --- Final Stage ---
FROM python:3.11-slim

WORKDIR /app

# copy only installed pacakges and source code
COPY --from=builder /app /app

CMD ["python", "fetch_data.py"]