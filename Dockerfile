FROM python:3.11-slim

ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app

RUN pip install --no-cache-dir "poetry==2.2.1"

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --with mlflow

COPY . .

CMD ["mlflow", "server"]
