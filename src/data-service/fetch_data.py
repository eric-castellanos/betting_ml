# fetch_data.py

import logging
from typing import Dict, List
#import requests
#import os

import polars as pl

from utils import save_data

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s")

logger = logging.getLogger(__name__)

def fetch_play_by_play_data(years: List[str], urls: List[str]) -> Dict[str, pl.DataFrame]:
    """
	Fetch NFL data from the given urls.
	Args:
        years (List[str]): List of years to get NFL data for.
		urls (List[str]): List of urls to fetch data from.
	Returns:
        Dictionary of polars dataframes.
	"""
    data_dict = {}

    for year, url in zip(years, urls):
        try:
            data_dict[f"play_by_play_{year}"] = pl.read_parquet(url)
        except OSError as e:
             logger.error(f"Network or file error for {url}: {e}")
        except ValueError as e:
             logger.error(f"Invalid parquet file or URL for {url}: {e}")
        except Exception as e:
            logger.error(f"Could not read in parquet file from {url} for {year}: {e}")
        else:
            logger.debug(f"Successfully read in play by play data for {year} from: {url}")

    return data_dict

# comment out this function for now, we don't need real time bet data atm
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

def main():
    # TODO: Set your API URL and parameters
    years = ["2020", "2021", "2022", "2023", "2024"]
    urls = [f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet" for year in years]
    pbp_data = fetch_play_by_play_data(years, urls)

    for year, df in pbp_data.items():
        save_data(df, bucket="sports-betting-ml", key="raw-data", filename=f"{year}_pbp_data.parquet", local=True, local_path="./data/raw")

    #odds_api_key = os.getenv("ODDS_API_KEY")
    #odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?apiKey={odds_api_key}&regions=us"
    #bet_data = fetch_bets_data(odds_url)

if __name__ == "__main__":
	main()
