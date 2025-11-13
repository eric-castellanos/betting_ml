# fetch_data.py

import logging
from typing import Dict, List
#import requests
#import os

import polars as pl
import click

from src.utils.utils import save_data

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s")

logger = logging.getLogger(__name__)

def fetch_play_by_play_data_logic(
    years: List[str],
    urls: List[str],
    local_path: str = "./data/raw",
    bucket: str = "sports-betting-ml",
    key: str = "raw-data",
    local: bool = True
) -> Dict[str, pl.DataFrame]:
    """
    Fetch NFL play-by-play data for specified years using plain Python logic.
    This function is used for testing and can also be used programmatically.
    """

    # Logic should NOT auto-fill URLs; tests expect validation to fail
    if not urls:
        raise ValueError("Number of years must match number of URLs.")

    # Input validation
    if len(years) != len(urls):
        raise ValueError("Number of years must match number of URLs.")

    for year in years:
        if not isinstance(year, str):
            raise ValueError("Year must be passed a string.")

    for url in urls:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL: URL must start with either http:// or https://.")

    data_dict: Dict[str, pl.DataFrame] = {}

    # Fetch data
    for year, url in zip(years, urls):
        try:
            df = pl.read_parquet(url)
            data_dict[f"play_by_play_{year}"] = df
            logger.info(f"Successfully read data for {year}")

            filename = f"{year}_pbp_data.parquet"
            save_data(df, bucket=bucket, key=key, filename=filename,
                      local=local, local_path=local_path)

        except OSError as e:
            logger.error(f"Network or file error for {url}: {e}")
            raise OSError(f"Network or file error for {url}") from e

        except ValueError as e:
            logger.error(f"Invalid parquet file for {url}: {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            raise

    return data_dict


# =====================================================================
#  CLICK COMMAND WRAPPER  (NEVER USED IN TESTS)
# =====================================================================
@click.command()
@click.option(
    "--years",
    multiple=True,
    default=["2020", "2021", "2022", "2023", "2024"],
    help="List of years to pull data for, e.g. --years 2020 2021 2022"
)
@click.option(
    "--urls",
    multiple=True,
    help="List of URLs to download data for. If omitted, defaults will be built from years."
)
@click.option(
    "--local-path",
    default="../../data/raw",
    help="Local folder to save data before uploading to S3."
)
@click.option(
    "--bucket",
    default="sports-betting-ml",
    help="S3 bucket to upload the data to."
)
@click.option(
    "--key",
    default="raw-data",
    help="S3 prefix (folder) under which data will be stored."
)
@click.option(
    "--local",
    default=True,
    help="Save locally if True, otherwise upload to S3 only."
)
def fetch_play_by_play_data(years, urls, local_path, bucket, key, local):
    """
    CLI wrapper that simply delegates to fetch_play_by_play_data_logic().
    """

    # Convert Click's tuple inputs → real lists
    years = list(years)
    urls = list(urls)

        # Build default URLs ONLY for CLI
    if not urls:
        urls = [
            f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
            for year in years
        ]

    result = fetch_play_by_play_data_logic(
        years=years,
        urls=urls,
        local_path=local_path,
        bucket=bucket,
        key=key,
        local=local
    )

    logger.info("Data fetching completed successfully.")
    return result


if __name__ == "__main__":
    fetch_play_by_play_data()
