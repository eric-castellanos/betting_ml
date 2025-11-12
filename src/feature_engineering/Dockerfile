# --- Builder Stage --- 
FROM python:3.10-slim AS builder

WORKDIR /app

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock ./

RUN poetry config virtualenvs.in-project true
RUN poetry install --with feature-engineering

# --- Final Stage ---
FROM python:3.10-slim AS final

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/feature_engineering/feature_engineering.py src/utils/utils.py ./
COPY ./data/raw ./data/raw

ENV PATH="./.venv/bin:$PATH"

CMD ["python", "feature_engineering.py", "--local", "True"]