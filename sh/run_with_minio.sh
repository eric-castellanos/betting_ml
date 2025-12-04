#!/usr/bin/env bash
set -euo pipefail

# Sync feature data from AWS S3 to MinIO, ensure MLflow artifact bucket exists,
# then run the XGBoost training script with MLflow pointing at MinIO.
#
# Override any of these via env vars before running.
AWS_PROFILE_SRC="${AWS_PROFILE_SRC:-default}"       # AWS profile with access to source data in AWS S3
AWS_PROFILE_MINIO="${AWS_PROFILE_MINIO:-minio}"     # AWS CLI profile configured for MinIO creds
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
FEATURE_BUCKET="${FEATURE_BUCKET:-sports-betting-ml}"
FEATURE_KEY="${FEATURE_KEY:-processed/features_2020-2024.parquet/features_2020-2024.parquet}"
ARTIFACT_BUCKET="${ARTIFACT_BUCKET:-mlflow-artifacts}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

echo "==> Copying features from AWS (${AWS_PROFILE_SRC}) to local temp"
aws --profile "$AWS_PROFILE_SRC" s3 cp "s3://${FEATURE_BUCKET}/${FEATURE_KEY}" "$tmp_file"

echo "==> Ensuring MinIO buckets exist"
aws --endpoint-url "$MINIO_ENDPOINT" --profile "$AWS_PROFILE_MINIO" s3 mb "s3://${FEATURE_BUCKET}" >/dev/null 2>&1 || true
aws --endpoint-url "$MINIO_ENDPOINT" --profile "$AWS_PROFILE_MINIO" s3 mb "s3://${ARTIFACT_BUCKET}" >/dev/null 2>&1 || true

echo "==> Uploading features to MinIO (${AWS_PROFILE_MINIO})"
aws --endpoint-url "$MINIO_ENDPOINT" --profile "$AWS_PROFILE_MINIO" s3 cp "$tmp_file" "s3://${FEATURE_BUCKET}/${FEATURE_KEY}"

echo "==> Running training with MLflow on MinIO"
AWS_PROFILE="$AWS_PROFILE_MINIO" \
AWS_ENDPOINT_URL="$MINIO_ENDPOINT" \
MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
MLFLOW_S3_ENDPOINT_URL="$MINIO_ENDPOINT" \
poetry run python -m src.model.xgboost \
  --mlflow \
  --bucket "$FEATURE_BUCKET" \
  --output-key-template "processed/features_2020-2024.parquet" \
  --filename "features_2020-2024.parquet" \
  --cutoff-season 2024

echo "==> Done"
