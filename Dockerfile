FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    psycopg2-binary \
    requests

CMD ["mlflow", "server"]
