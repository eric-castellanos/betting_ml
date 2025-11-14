"""Script to fetch raw pbp and bet data"""

import logging
from typing import Dict, List
#import requests
#import os

import polars as pl
import click

from src.utils.utils import save_data

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s"
)

logger = logging.getLogger(__name__)

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
    help="S3 prefix (folder) to upload data under."
)
@click.option(
    "--local",
    default=True,
    help="Boolean flag to determine whether to save locally or to S3."
)
def fetch_play_by_play_data(years: List[str], urls: List[str], local_path: str, bucket: str, key: str, local: bool) -> Dict[str, pl.DataFrame]:
    """
    Fetch NFL play-by-play data for specified years from given URLs, save each year's data locally and/or upload to S3.

    Args:
        years (List[str]): List of years to fetch data for (e.g., ["2020", "2021"]).
        urls (List[str]): List of URLs to download Parquet files from. If empty, defaults are built from years.
        local_path (str): Local directory to save Parquet files.
        bucket (str): S3 bucket name to upload files to.
        key (str): S3 prefix (folder) to upload files under.
        local (bool): If True, saves files locally in addition to S3 upload.

    Returns:
        Dict[str, pl.DataFrame]: Dictionary mapping year keys (e.g., "play_by_play_2020") to Polars DataFrames.

    Raises:
        ValueError: If input validation fails (e.g., mismatched years/urls, invalid year or URL).
        OSError: If there is a network or file error when reading Parquet files.
        Exception: For any other unexpected errors during data fetching or saving.
    """
    # ✅ Build default URLs if none provided
    if not urls:
        urls = [
            f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet"
            for year in years
        ]

    if len(years) != len(urls):
        raise ValueError("Number of years must match number of URLs.")

    for year in years:
        if not isinstance(year, str):
            raise ValueError("Year must be passed a string.")

    for url in urls:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL: URL must start with either http:// or https://.")

    data_dict = {}

    for year, url in zip(years, urls):
        try:
            df = pl.read_parquet(url)
            data_dict[f"play_by_play_{year}"] = df
            logger.info(f"✅ Successfully read data for {year}")

            # Save locally and to S3
            filename = f"{year}_pbp_data.parquet"
            save_data(df, bucket=bucket, key=key, filename=filename, local=local, local_path=local_path)

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

# comment out this function for now, 
# we don't need real time bet data atm
# def fetch_bets_data(url : str) -> Dict[str, Any]:
#     """
#     Args:
#         url: bet url from odds-api.
#     """
#     try:
#         response = requests.get(url)
#     except requests.exceptions.RequestException as e:
#         logger.error(f"Reuqest to {url} failed")
#     else:
#         return response.json()

if __name__ == "__main__":
    #odds_api_key = os.getenv("ODDS_API_KEY")
    #odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?apiKey={odds_api_key}&regions=us"
    #bet_data = fetch_bets_data(odds_url)

	fetch_play_by_play_data()
