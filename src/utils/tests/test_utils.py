"""
Unit tests for src/utils/utils.py
"""
import pytest
import polars as pl
from unittest.mock import patch, MagicMock
import os
from src.utils import utils

def test_save_data_local(tmp_path):
    """
    Test saving a Polars DataFrame locally using save_data.
    """
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    filename = "test.parquet"
    local_path = tmp_path
    utils.save_data(df, filename=filename, local=True, local_path=str(local_path))
    assert os.path.exists(os.path.join(local_path, filename))

@patch("src.utils.utils._s3_client")
def test_save_data_s3_skips_if_exists(mock_s3_client):
    """
    Test that save_data skips S3 upload if file exists in bucket.
    """
    df = pl.DataFrame({"a": [1, 2]})
    mock_client = MagicMock()
    mock_client.head_object.return_value = True
    mock_s3_client.return_value = mock_client
    utils.save_data(df, bucket="bucket", key="key", filename="file.parquet", local=False)
    mock_client.head_object.assert_called_once()

@patch("src.utils.utils._s3_client")
def test_save_data_s3_uploads_if_not_exists(mock_s3_client):
    """
    Test that save_data uploads to S3 if file does not exist.
    """
    df = pl.DataFrame({"a": [1, 2]})
    mock_client = MagicMock()
    from botocore.exceptions import ClientError
    mock_client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "head_object")
    mock_client.upload_file = MagicMock()
    mock_s3_client.return_value = mock_client
    utils.save_data(df, bucket="bucket", key="key", filename="file.parquet", local=False)
    mock_client.upload_file.assert_called()


def test_load_data_requires_filename():
    with pytest.raises(ValueError):
        utils.load_data(local=True, local_path=".")


def test_load_data_local_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        utils.load_data(filename="missing.parquet", local=True, local_path=str(tmp_path))
