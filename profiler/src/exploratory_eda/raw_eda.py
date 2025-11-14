import logging
import sys
from datetime import datetime
import os

from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import matplotlib.font_manager as fm
import polars as pl
import pandas as pd
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.utils.utils import save_data, load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_raw_eda_report(raw_bucket: str, raw_key: str, report_bucket: str, report_key: str, local: bool = True, local_path: str = "/tmp"):
    """
    Load raw data from S3, generate an EDA report, and save it locally and/or to S3.
    Args:
        bucket (str): S3 bucket for raw data.
        key (str): S3 key for raw data file.
        report_bucket (str): S3 bucket for saving the report.
        report_key (str): S3 key (folder) for saving the report.
        local (bool): Whether to save the report locally.
        local_path (str): Local directory to save the report.
    Returns:
        Tuple of (local_path, filename)
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    report_name = raw_key.split("/")[-1].replace(".parquet", "")
    filename = f"{report_name}_eda_report_{today_str}.html"
    local_file = os.path.join(local_path, filename)

    try:
        logging.info(f"Loading raw data from S3: s3://{raw_bucket}/{raw_key}")
        # loaded object should always be a polars dataframe
        df = load_data(raw_bucket, raw_key).to_pandas()

        logging.info(f"Generating raw data EDA report at {local_file}")
        report = ProfileReport(
            df,
            title="NFL Play by Play Raw Data EDA Report",
            minimal=True,
            explorative=False
        )
        report.to_file(local_file)
        logging.info("Raw data EDA report generation complete")

        # Save report to S3
        with open(local_file, "rb") as f:
            report_data = f.read()
        save_data(
            data=pl.DataFrame(),  # Dummy DataFrame, not used for HTML
            bucket=report_bucket,
            key=report_key,
            filename=filename,
            local=local,
            local_path=local_path
        )
        # Actually upload the HTML file to S3
        s3_client = boto3.client("s3")
        s3_client.put_object(Bucket=report_bucket, Key=f"{report_key}/{filename}", Body=report_data, ContentType="text/html")

        return local_file, filename
    except Exception as e:
        logging.exception("Failed to generate raw data EDA report")
        raise

if __name__ == "__main__":
    generate_raw_eda_report(raw_bucket="sports-betting-ml", raw_key="raw-data/2020_pbp_data.parquet", report_bucket="sports-betting-ml", report_key="eda/reports", local=True)
    generate_raw_eda_report(raw_bucket="sports-betting-ml", raw_key="raw-data/2021_pbp_data.parquet", report_bucket="sports-betting-ml", report_key="eda/reports", local=True)
    generate_raw_eda_report(raw_bucket="sports-betting-ml", raw_key="raw-data/2022_pbp_data.parquet", report_bucket="sports-betting-ml", report_key="eda/reports", local=True)
    generate_raw_eda_report(raw_bucket="sports-betting-ml", raw_key="raw-data/2023_pbp_data.parquet", report_bucket="sports-betting-ml", report_key="eda/reports", local=True)
    generate_raw_eda_report(raw_bucket="sports-betting-ml", raw_key="raw-data/2024_pbp_data.parquet", report_bucket="sports-betting-ml", report_key="eda/reports", local=True)