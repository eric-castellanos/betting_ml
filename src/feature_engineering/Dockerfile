FROM python:3.11-slim AS builder

WORKDIR /app

COPY ./pyproject.toml ./poetry.lock ./

RUN pip install poetry

RUN poetry config virtualenvs.in-project true
RUN poetry install --with feature-engineering

### --- Final Stage --- ###
FROM python:3.11-slim AS final

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY ./src/feature_engineering/feature_engineering.py ./src/utils/utils.py ./

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "feature_engineering.py", "--local", "--input-path", "../../data/raw", "--filename", "2020_pbp_data.parquet", "--output-path", "../../data/processed"]
