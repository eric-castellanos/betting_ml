"""Fetch NFL spread odds from The Odds API and save them locally."""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import click
import requests

from src.utils.utils import save_data


API_URL_TEMPLATE = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT = "americanfootball_nfl"
MARKETS = ["spreads"]
TIMEOUT = 30
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "odds"
DEFAULT_SEASON_YEAR = datetime.now().year

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - Line Number %(lineno)d - %(message)s"
)

logger = logging.getLogger(__name__)


def fetch_odds(api_key: str, sport: str, markets: list[str], timeout: int) -> Any:
    """Pull spread odds and return the JSON payload."""
    params = {"apiKey": api_key, "markets": ",".join(markets)}
    url = API_URL_TEMPLATE.format(sport=sport)

    try:
        logger.info(
            "Requesting odds data",
            extra={"url": url, "sport": sport, "markets": params["markets"]},
        )
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:  # type: ignore[catching-any]
        logger.exception("Request to The Odds API failed")
        sys.exit(1)

    if response.status_code != 200:
        logger.error(
            "Non-200 response from The Odds API",
            extra={"status_code": response.status_code, "body": response.text},
        )
        sys.exit(1)

    payload = response.json()
    if isinstance(payload, list):
        logger.info("Fetched odds payload", extra={"entries": len(payload)})
    else:
        logger.info("Fetched odds payload (non-list response)")

    return payload


def _labor_day(season_year: int) -> datetime:
    """Return the date of Labor Day (first Monday of September) for a given year."""
    sept_first = datetime(season_year, 9, 1)
    days_until_monday = (0 - sept_first.weekday()) % 7
    return sept_first + timedelta(days=days_until_monday)


def compute_nfl_week(today: Optional[datetime] = None, season_year: Optional[int] = None) -> tuple[int, int]:
    """
    Compute the NFL season year and week number based on current date.

    Week 1 starts the Thursday after Labor Day. Weeks increment every 7 days.
    """
    current_date = today or datetime.now()
    season = season_year or current_date.year
    labor_day = _labor_day(season)
    season_start = labor_day + timedelta(days=3)  # Thursday after Labor Day

    if current_date < season_start:
        season -= 1
        labor_day = _labor_day(season)
        season_start = labor_day + timedelta(days=3)

    delta_days = (current_date - season_start).days
    week = (delta_days // 7) + 1
    week = max(1, min(week, 22))  # cap at typical NFL season length (incl. playoffs buffer)

    logger.debug(
        "Computed NFL week",
        extra={
            "season_year": season,
            "week": week,
            "today": current_date.isoformat(),
            "season_start": season_start.date().isoformat(),
        },
    )
    return season, week


def build_output_filename(season_year: int, week: int) -> str:
    """Compose an odds filename with season and week."""
    return f"odds_{season_year}_week{week:02d}.json"


def main(
    api_key: Optional[str] = None,
    timeout: int = TIMEOUT,
    bucket: Optional[str] = None,
    key: Optional[str] = None,
    local: bool = True,
    local_output_dir: Path = OUTPUT_DIR,
    season_year: Optional[int] = None,
    week: Optional[int] = None,
) -> None:
    """Entry point to fetch odds using environment defaults."""
    resolved_api_key = api_key or os.getenv("ODDS_API_KEY")
    if not resolved_api_key:
        logger.error("ODDS_API_KEY environment variable is not set.")
        sys.exit(1)

    resolved_year, resolved_week = compute_nfl_week(season_year=season_year)
    if week is not None:
        resolved_week = week
    output_filename = build_output_filename(resolved_year, resolved_week)
    payload = fetch_odds(resolved_api_key, SPORT, MARKETS, timeout)

    if local:
        save_data(
            payload,
            filename=output_filename,
            local=True,
            local_path=str(local_output_dir),
            data_format="json",
        )
    if bucket:
        key_prefix = key if key is not None else None
        save_data(
            payload,
            bucket=bucket,
            key=key_prefix,
            filename=output_filename,
            local=False,
            data_format="json",
        )


@click.command()
@click.option(
    "--timeout",
    default=TIMEOUT,
    show_default=True,
    help="HTTP request timeout in seconds.",
)
@click.option(
    "--bucket",
    default=None,
    help="S3 bucket to upload the odds JSON. If omitted, skip upload.",
)
@click.option(
    "--key",
    default=None,
    show_default=True,
    help="S3 key prefix for the odds JSON (defaults to the filename if omitted).",
)
@click.option(
    "--local/--no-local",
    default=True,
    show_default=True,
    help="Write the odds JSON locally.",
)
@click.option(
    "--local-output-path",
    type=click.Path(path_type=Path),
    default=OUTPUT_DIR,
    show_default=True,
    help="Directory to write the local odds.json file.",
)
@click.option(
    "--season-year",
    type=int,
    default=None,
    help="Season year for the odds filename (defaults to current year).",
)
@click.option(
    "--week",
    type=int,
    default=None,
    help="Week number for the odds filename (defaults to current week number).",
)
def cli(
    timeout: int,
    bucket: Optional[str],
    key: Optional[str],
    local: bool,
    local_output_path: Path,
    season_year: Optional[int],
    week: Optional[int],
) -> None:
    """CLI wrapper for fetching odds data."""
    main(
        api_key=None,
        timeout=timeout,
        bucket=bucket,
        key=key,
        local=local,
        local_output_dir=local_output_path,
        season_year=season_year,
        week=week,
    )


if __name__ == "__main__":
    cli()
