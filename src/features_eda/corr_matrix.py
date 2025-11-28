"""
Generate and save a correlation matrix heatmap for the features dataset.
"""

import logging
import os
import sys
from datetime import datetime

import boto3
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd

from src.utils.utils import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_corr_matrix(
    load_file: str = "features_2020-2024.parquet",
    features_bucket: str = "sports-betting-ml",
    features_key: str = "processed/features_2020-2024.parquet",
    output_bucket: str = "sports-betting-ml",
    output_key: str = "eda/corr_matrix",
    local: bool = True,
    local_path: str = "/tmp",
) -> tuple[str, str]:
    """
    Load features data from S3, build a correlation matrix heatmap with matplotlib,
    save locally, and upload to S3 under the specified key.
    """
    today_str = datetime.today().strftime("%Y-%m-%d")
    filename = f"corr_matrix_{today_str}.png"
    local_file = os.path.join(local_path, filename)

    # Load features (polars) then convert to pandas for plotting
    logger.info("Loading features data from S3 for correlation matrix: s3://%s/%s/%s", features_bucket, features_key, load_file)
    features_df = load_data(filename=load_file, bucket=features_bucket, key=features_key)
    pdf = features_df.to_pandas() if isinstance(features_df, pl.DataFrame) else pd.DataFrame(features_df)

    num_df = pdf.select_dtypes(include="number")
    corr = num_df.corr()

    logger.info("Creating correlation matrix plot")
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.index)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs(local_path, exist_ok=True)
    plt.savefig(local_file, dpi=300)
    plt.close(fig)
    logger.info("Saved correlation matrix locally at %s", local_file)

    s3_key = f"{output_key.rstrip('/')}/{filename}"
    logger.info("Uploading correlation matrix to s3://%s/%s", output_bucket, s3_key)
    s3_client = boto3.client("s3")
    with open(local_file, "rb") as f:
        s3_client.put_object(Bucket=output_bucket, Key=s3_key, Body=f, ContentType="image/png")

    if not local:
        try:
            os.remove(local_file)
        except OSError:
            pass

    return local_file, s3_key


if __name__ == "__main__":
    generate_corr_matrix()
