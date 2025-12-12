# NFL Spread Prediction

ML stack for predicting NFL point spreads. Trains an XGBoost model on play-by-play data, tracks experiments with MLflow, and serves real-time predictions through a FastAPI inference service. Docker Compose brings up Postgres + MinIO for MLflow’s backend/artifacts, an MLflow tracking server, and the inference API that pulls the latest registered model.

## Components
- `mlflow`: tracking server backed by Postgres (backend store) and MinIO (artifact store)
- `inference`: FastAPI service exposing `/`, `/health`, and `/predict` (reads `MLFLOW_MODEL_URI`)
- Pipelines for data ingestion (`data_service`), feature engineering, and model training (`ml-model`)
- Sample request payloads (`api_examples/`) and analysis artifacts (`artifacts/`)

## Configuration
Set environment variables in `.env`; they are passed into services by docker-compose:
- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_HOST`, `POSTGRES_PORT`
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
- `MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `MLFLOW_BACKEND_URI`, `MLFLOW_ARTIFACT_ROOT`
- `MLFLOW_MODEL_URI` for the inference service (defaults to `models:/nfl_spread_model/1`)

## Run the stack
```bash
docker-compose up --build
```

The MLflow image installs only the `[tool.poetry.group.mlflow.dependencies]` group via:
```bash
poetry install --no-root --with mlflow
```
