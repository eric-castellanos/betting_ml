from typing import Optional
import logging
import os
import tempfile

import polars as pl
from boto3.session import Session
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s")

logger = logging.getLogger(__name__)


def _s3_client():
    """
    Build an S3 client honoring AWS_ENDPOINT_URL (for MinIO) if set.
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    session = Session()
    return session.client("s3", endpoint_url=endpoint_url)

def save_data(data: pl.DataFrame, bucket : Optional[str] = None, key : Optional[str] = None, filename: Optional[str] = None, local: bool = False, local_path : Optional[str] = None) -> None:
    """
    Save data to S3 bucket. If local set to True, save data to a local parquet file.
    Args:
		data (dict): Data to save.
        bucket (str): S3 bucket.
        key (str): file path inside S3 bucket.
		filename (str): Path to the output file.
		local (bool): Bool to determine whether or not to save locally.
        local_path (str): local filepath to save data to.
	"""
    if not filename:
        logger.error("Filename must be provided.")
        return

    s3_client = _s3_client()
    s3_key = f"{key}/{filename}"

    if local and local_path:
        full_path = os.path.join(local_path, filename)
        if os.path.exists(full_path):
            logger.info(f"Skipping local save: {full_path} already exists.")
        else:
            os.makedirs(local_path, exist_ok=True)
            logger.info(f"Writing {filename} to local folder: {local_path}")
            data.write_parquet(full_path)

    
    if not local:
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            logger.info(f"Skipping S3 upload: s3://{bucket}/{s3_key} already exists.")
            return  # File exists, skip upload
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # Object not found — proceed to upload
                pass
            else:
                logger.error(f"Error checking S3 for {s3_key}: {e}")
                raise

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
            data.write_parquet(tmp_path)
            logger.info(f"Uploading new file to s3://{bucket}/{s3_key}")
            s3_client.upload_file(tmp_path, bucket, s3_key)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

def load_data(
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    filename: Optional[str] = None,
    local: bool = False,
    local_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load data from S3 or local filesystem.
    Args:
        bucket (str): S3 bucket name.
        key (str): Path inside the S3 bucket.
        filename (str): Name of the file to load.
        local (bool): If True, load from local_path instead of S3.
        local_path (str): Local directory containing the file.
    Returns:
        pl.DataFrame: Loaded Polars DataFrame.
    """

    if not filename:
        raise ValueError("filename must be provided to load_data().")

    # ----------------------------------
    # LOCAL READ
    # ----------------------------------
    if local:
        if not local_path:
            raise ValueError("local_path must be provided when local=True.")
        
        full_path = os.path.join(local_path, filename)

        if not os.path.exists(full_path):
            logger.error(f"Local file does not exist: {full_path}")
            raise FileNotFoundError(f"Local file not found: {full_path}")

        logger.info(f"Loading local file: {full_path}")
        try:
            return pl.read_parquet(full_path)
        except Exception as e:
            logger.error(f"Failed to read local parquet file {full_path}: {e}")
            raise

    # ----------------------------------
    # S3 READ
    # ----------------------------------
    s3_client = _s3_client()
    s3_key = f"{key}/{filename}"

    tmp_path = None
    try:
        logger.info(f"Downloading file from s3://{bucket}/{s3_key}")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        s3_client.download_file(bucket, s3_key, tmp_path)
        df = pl.read_parquet(tmp_path)
        logger.info(f"Loaded s3://{bucket}/{s3_key} successfully")
        return df

    except ClientError as e:
        logger.error(f"Failed to download s3://{bucket}/{s3_key}: {e}")
        raise

    except Exception as e:
        logger.error(f"Failed to read parquet from s3://{bucket}/{s3_key}: {e}")
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def polars_info(df: pl.DataFrame) -> pl.DataFrame:
    """
    Returns a summary DataFrame with column names, dtypes, and null counts.
    """
    return pl.DataFrame({
        "column": list(df.schema.keys()),
        "dtype": [str(v) for v in df.schema.values()],
        "nulls": [df[col].null_count() for col in df.columns],
        "n_rows": [df.height] * len(df.columns),
        "n_cols": [df.width] * len(df.columns),
    })
