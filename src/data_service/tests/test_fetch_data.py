# pylint: disable=C0301
"""Unit test file for fetch_data.py file"""
from unittest.mock import patch

import pytest
import polars as pl

from src.data_service.fetch_data import fetch_play_by_play_data

@pytest.mark.parametrize(
    "years, urls, expected_keys",
    [
        (
            ["2020"],
            ["https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2020.parquet"],
            ["play_by_play_2020"]
        ),
        (
            ["2021", "2022"],
            [
                "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2021.parquet",
                "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2022.parquet"
            ],
            ["play_by_play_2021", "play_by_play_2022"]
        ),
    ]
)
def test_fetch_play_by_play_data_valid(years: list[str], urls: list[str], expected_keys: list[str]) -> None:
    """
    Test that fetch_play_by_play_data returns a dictionary of polars DataFrames
    for valid years and urls, and that the keys and shapes are as expected.
    Uses mocking to avoid real network calls.

    Args:
        years (list[str]): List of year strings to fetch data for.
        urls (list[str]): List of URLs corresponding to each year.
        expected_keys (list[str]): List of expected dictionary keys for the result.
    """
    dummy_df = pl.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with patch("polars.read_parquet", return_value=dummy_df):
        result = fetch_play_by_play_data(years, urls)
        assert isinstance(result, dict)
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], pl.DataFrame)
            assert result[key].shape == dummy_df.shape


@pytest.mark.parametrize(
    "years, urls, exception, message",
    [
        (["2020"], [], ValueError, "Number of years must match number of URLs."),
        ([2020], ["url"], ValueError, "Year must be passed a string."),
        (["2020"], ["not_a_url"], ValueError, "Invalid URL: URL must start with either http:// or https://.")
    ]
)
def test_fetch_play_by_play_data_invalid(years: list, urls: list, exception: type[Exception], message: str) -> None:
    """
    Test that fetch_play_by_play_data raises the correct exception and error message
    for invalid input cases, such as mismatched lengths, non-string years, or malformed URLs.

    Args:
        years (list): List of year values (may be invalid type for test).
        urls (list): List of URLs (may be empty or malformed for test).
        exception (type[Exception]): Exception type expected to be raised.
        message (str): Expected error message substring.
    """
    with pytest.raises(exception) as excinfo:
        fetch_play_by_play_data(years, urls)
    assert message in str(excinfo.value)

def test_fetch_play_by_play_data_oserror() -> None:
    """
    Test that fetch_play_by_play_data raises an OSError when pl.read_parquet fails,
    simulating a network or file error using mocking.

    Args:
        None
    """
    with patch("src.data_service.fetch_data.pl.read_parquet", side_effect=OSError("mocked failure")):
        years = ["2020"]
        urls = ["http://valid-url.parquet"]

        with pytest.raises(OSError) as excinfo:
            fetch_play_by_play_data(years, urls)

        assert "Network or file error for http://valid-url.parquet" in str(excinfo.value)
