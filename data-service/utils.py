from typing import Optional
import logging

import polars as pl
import boto3
import io

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levlname)s - Line Number %(lineno)d")

logger = logging.getLogger(__name__)

def save_data(data: pl.DataFrame, bucket : str, key : str, filename: Optional[str], local: bool = False) -> None:
    """
    Save data to S3 bucket. If local set to True, save data to a local parquet file.
    Args:
		data (dict): Data to save.
		filename (str): Path to the output file.
		local (bool): Bool to determine whether or not to save locally.
	"""
    if local:
        logger.debug(f"Writing {filename} to the local data folder.")
        data.write_parquet(filename)

    client = boto3.client("s3")
    buffer = io.BytesIO()

    logger.debug(f"Uploading file to S3 at path: s3://{bucket}/{key}")
    client.upload_fileobj(buffer, bucket, key)