from typing import Optional
import logging
import io

import polars as pl
import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s")

logger = logging.getLogger(__name__)

def save_data(data: pl.DataFrame, bucket : str, key : str, filename: Optional[str] = None, local: bool = False, local_path : Optional[str] = None) -> None:
    """
    Save data to S3 bucket. If local set to True, save data to a local parquet file.
    Args:
		data (dict): Data to save.
		filename (str): Path to the output file.
		local (bool): Bool to determine whether or not to save locally.
	"""
    if local:
        if not filename or not local_path:
            logger.error("filename and local_path must be provided for local save.")
        else:
            full_path = f"{local_path}/{filename}"
            logger.info(f"Writing {filename} to the local data folder.")
            data.write_parquet(full_path)

    buffer = io.BytesIO()
    data.write_parquet(buffer)
    buffer.seek(0)

    client = boto3.client("s3")
    logger.info(f"Uploading file to S3 at path: s3://{bucket}/{key}")
    client.upload_fileobj(buffer, bucket, f"{key}/{filename}")