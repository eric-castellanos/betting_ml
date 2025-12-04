FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-root --with endpoint

# Copy only what is needed for the endpoint
COPY src/__init__.py ./src/__init__.py
COPY src/inference_service ./src/inference_service

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.inference_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
