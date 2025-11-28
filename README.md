## MLflow stack

Services (docker-compose):
- PostgreSQL backend store
- MinIO artifact store
- MLflow tracking server

Configuration is driven by `.env`; update credentials/URIs there. docker-compose loads `.env` and passes all values into services. Key vars:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
- `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `MLFLOW_BACKEND_URI`, `MLFLOW_ARTIFACT_ROOT`

Bring the stack up:
```bash
docker-compose up --build
```

MLflow image is built with Poetry. MLflow dependencies live under `[tool.poetry.group.mlflow.dependencies]` in `pyproject.toml`; the Dockerfile runs:
```
poetry install --no-root --with mlflow
```
to install only that group for the container.
