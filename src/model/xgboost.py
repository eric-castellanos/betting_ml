"""
Helpers for fetching feature data for the xgboost model.
"""

from typing import Optional

import polars as pl

from src.utils.utils import load_data


def load_feature_dataset(
    year: int = 2020,
    bucket: str = "sports-betting-ml",
    output_key_template: str = "processed/features_{year}.parquet",
    filename: str = "2020_pbp_data.parquet",
    local: bool = False,
    local_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load the processed feature dataset that was written by the feature_engineering CLI.

    The defaults mirror the CLI options shown in feature_engineering.py so this will
    load from s3://sports-betting-ml/processed/features_{year}.parquet/<filename>.
    Set local=True and provide local_path to read from disk instead.
    """
    key = output_key_template.format(year=year)
    return load_data(bucket=bucket, key=key, filename=filename, local=local, local_path=local_path)


if __name__ == "__main__":
    features_df = load_feature_dataset()
    import pdb; pdb.set_trace()
