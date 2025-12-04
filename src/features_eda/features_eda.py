import logging
import sys
from datetime import datetime
import os

from ydata_profiling import ProfileReport
import polars as pl
import boto3

from src.utils.utils import save_data, load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def generate_features_eda_report(
    load_file: str = "features_2020-2024.parquet",
    features_bucket: str = "sports-betting-ml",
    features_key: str = "processed/features_2020-2024.parquet",
    report_bucket: str = "sports-betting-ml",
    report_key: str = "eda/reports",
    local: bool = True,
    local_path: str = "/tmp",
):
    """
    Load featurees data from S3, generate an EDA report, and save it locally and/or to S3.
    Args:
        features_bucket (str): S3 bucket for raw data.
        features_key (str): S3 key for raw data file.
        report_bucket (str): S3 bucket for saving the report.
        report_key (str): S3 key (folder) for saving the report.
        local (bool): Whether to save the report locally.
        local_path (str): Local directory to save the report.
    Returns:
        Tuple of (local_path, filename)
    """
    today_str = datetime.today().strftime('%Y-%m-%d')
    report_name = load_file.replace(".parquet", "")
    filename = f"{report_name}_features_eda_report_{today_str}.html"
    local_file = os.path.join(local_path, filename)

    try:
        logging.info(f"Loading features data from S3: s3://{features_bucket}/{features_key}/{load_file}")
        # loaded object should always be a polars dataframe
        df = load_data(filename=load_file, bucket=features_bucket, key=features_key).to_pandas()

        logging.info(f"Generating features data EDA report at {local_file}")
        report = ProfileReport(
            df,
            title="NFL Play by Play Features Data EDA Report",
            minimal=False,
            explorative=False,
            correlations={
                "pearson": {"calculate": True}
            }
        )
        report.to_file(local_file)
        logging.info("Features data EDA report generation complete")

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
    generate_features_eda_report()
