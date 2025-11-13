from typing import Optional
import logging
import io
import os

import polars as pl
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s")

logger = logging.getLogger(__name__)

def save_data(data: pl.DataFrame, filename: Optional[str] = None, local: bool = False, local_path : str = "./data", bucket: Optional[str] = None, key: Optional[str] = None) -> None:
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

    if local:
        full_path = os.path.join(local_path, filename)
        if os.path.exists(full_path):
            logger.info(f"Skipping local save: {full_path} already exists.")
        else:
            os.makedirs(local_path, exist_ok=True)
            logger.info(f"Writing {filename} to local folder: {local_path}")
            data.write_parquet(full_path)

    
    else:
        try:
            s3_client = boto3.client("s3")
            s3_key = f"{key}/{filename}"

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

        buffer = io.BytesIO()
        data.write_parquet(buffer)
        buffer.seek(0)
        logger.info(f"Uploading new file to s3://{bucket}/{s3_key}")
        s3_client.upload_fileobj(buffer, bucket, s3_key)

def load_data(local: bool = False, filename: Optional[str] = None, local_path: str = "./data", bucket: Optional[str] = None, key: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Load a Parquet file from S3 or locally and return as a Polars DataFrame.
    Args:
        bucket (str): S3 bucket name.
        key (str): S3 object key (path to the file).
        local (bool): If True, load from local file instead of S3.
        filename (str): Local filename to load (required if local=True).
        local_path (str): Local directory to load file from.
    Returns:
        pl.DataFrame or None: Loaded DataFrame, or None if load fails.
    Raises:
        botocore.exceptions.ClientError: If S3 access fails.
        OSError: If file cannot be read as Parquet.
    """
    if local:
        if not filename:
            logger.error("Filename must be provided for local load.")
            return None
        local_file = os.path.join(local_path, filename)
        try:
            df = pl.read_parquet(local_file)
            logger.info(f"Loaded file from local path: {local_file}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet from local path {local_file}: {e}")
            raise
    
    else:
        # S3 load (default)
        s3_client = boto3.client("s3")
        buffer = io.BytesIO()
        try:
            s3_client.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            df = pl.read_parquet(buffer)
            logger.info(f"Loaded file from s3://{bucket}/{key}")
            return df
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Parquet from s3://{bucket}/{key}: {e}")
            raise

def polars_info(df: pl.DataFrame):
    info = df.select([
        pl.lit(df.height).alias("n_rows"),
        pl.lit(df.width).alias("n_cols"),
    ])
    null_counts = df.null_count().transpose(include_header=True)
    schema = pl.DataFrame({
        "column": list(df.schema.keys()),
        "dtype": [str(v) for v in df.schema.values()],
        "nulls": [df[col].null_count() for col in df.columns],
    })
    return schema