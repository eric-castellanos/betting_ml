# fetch_data.py

import requests
import os
import logging
from typing import Dict, List, Any

import polars as pl

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levlname)s - Line Number %(lineno)d")

logger = logging.getLogger(__name__)

def fetch_play_by_play_data(years: List[str], urls: List[str]) -> Dict[str, pl.DataFrame]:
    """
	Fetch NFL data from the given API URL.
	Args:
		url (str): The url to fetch data from.
	Returns:
        Dictionary of polars dataframes
	"""
    data_dict = {}

    for year, url in zip(years, urls):
        try:
            data_dict[f"play_by_play_{year}"] = pl.read_parquet(url)
        except Exception as e:
            logger.error(f"Could not read in parquet file from {url} for {year}: {e}")
        else:
            logger.debug(f"Successfully read in play by play data for {year} from: {url}")

    return data_dict

def save_data(data: pl.DataFrame, filename: str, local: bool = False) -> None:
	"""
	Save data to a local JSON file.
	Args:
		data (dict): Data to save.
		filename (str): Path to the output file.
		local (bool): Bool to determine whether or not to save locally.
	"""
	if local:
		data.write_parquet(filename)

def main():
	# TODO: Set your API URL and parameters
	years = ["2020", "2021", "2022", "2023", "2024"]
	urls = [f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet" for year in years]
	data = fetch_play_by_play_data(years, urls)
	#save_data(data, "nfl_data.json")

if __name__ == "__main__":
	main()
