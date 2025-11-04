### --- Builder Stage --- ###
FROM python:3.10-slim AS builder

WORKDIR /app

COPY ../../pyproject.toml ../../poetry.lock ./

RUN pip install poetry

RUN poetry config virtualenvs.in-project true
RUN poetry install --with fetch-data

### --- Final Stage --- ###
FROM python:3.10-slim AS final

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY ./fetch_data.py ./utils.py ./

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "fetch_data.py"]

