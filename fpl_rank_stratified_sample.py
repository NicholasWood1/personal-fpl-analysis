"""Build a rank-stratified sample of FPL managers and their GW histories.

The script reads the current FPL manager population, samples standings pages
across rank bands, fetches histories for the configured manager subset, and
saves resumable raw history and tidy Parquet datasets under ./data/.

After checking the pilot, set HISTORY_MANAGER_LIMIT = None to fetch histories
for the rest of the 10,000-manager sample. The JSONL cache means completed
manager history requests will not be repeated.
"""

from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://fantasy.premierleague.com/api/"
OVERALL_LEAGUE_ID = 314
SEASON = "2026-27"

TARGET_MANAGER_SAMPLE = 50
MANAGERS_PER_PAGE = 50
RANDOM_SEED = 2026

# Use a small sample for the test run; change to None to fetch all sampled
# managers after verifying the data and request behaviour.
HISTORY_MANAGER_LIMIT: int | None = 1

# A small pause between requests. The FPL API does not publish a rate limit.
REQUEST_DELAY_SECONDS = 0.25

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

STANDINGS_FILE = RAW_DIR / f"standings_sample_{SEASON}.parquet"
HISTORY_CACHE_FILE = RAW_DIR / f"manager_history_raw_{SEASON}.jsonl"
HISTORY_FILE = PROCESSED_DIR / f"manager_gameweek_history_{SEASON}.parquet"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    """Create a session with retries for temporary API/network errors."""
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            # Replace the contact placeholder with your email address.
            "User-Agent": "personal-fpl-analysis/0.1 (contact: mlg.wood1@gmail.com)"
        }
    )
    return session


def get_json(
    session: requests.Session,
    endpoint: str,
    params: dict | None = None,
) -> dict:
    """GET an FPL API endpoint and return its JSON response."""
    response = session.get(
        f"{BASE_URL}{endpoint.lstrip('/')}",
        params=params,
        timeout=(10, 30),
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Rank-stratified standings sample
# ---------------------------------------------------------------------------

def choose_standings_pages(
    total_pages: int,
    target_managers: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Choose one random page from each equal-sized rank band.

    Returns (stratum_number, page_number) pairs. With a 10,000-manager
    target and 50 managers per page, this selects 200 pages.
    """
    pages_needed = min(
        math.ceil(target_managers / MANAGERS_PER_PAGE),
        total_pages,
    )
    rng = random.Random(seed)
    selected_pages: list[tuple[int, int]] = []

    for stratum in range(pages_needed):
        first_page = math.floor(stratum * total_pages / pages_needed) + 1
        last_page = math.floor((stratum + 1) * total_pages / pages_needed)
        if last_page < first_page:
            continue

        page = rng.randint(first_page, last_page)
        selected_pages.append((stratum + 1, page))

    return selected_pages


def collect_standings_sample(session: requests.Session) -> pd.DataFrame:
    """Download selected pages from the Overall league standings."""
    print("Fetching FPL bootstrap data...")
    bootstrap = get_json(session, "bootstrap-static/")

    total_managers = int(bootstrap["total_players"])
    total_pages = math.ceil(total_managers / MANAGERS_PER_PAGE)
    finished_events = [
        int(event["id"])
        for event in bootstrap.get("events", [])
        if event.get("finished")
    ]
    latest_finished_gw = max(finished_events, default=None)

    selected_pages = choose_standings_pages(
        total_pages=total_pages,
        target_managers=TARGET_MANAGER_SAMPLE,
        seed=RANDOM_SEED,
    )

    print(f"FPL manager population reported by API: {total_managers:,}")
    print(f"Standings pages in estimated population: {total_pages:,}")
    print(f"Selected pages: {len(selected_pages):,}")
    print(f"Latest gameweek marked finished: {latest_finished_gw}")

    snapshot_utc = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    for stratum, page in tqdm(selected_pages, desc="Standings pages"):
        payload = get_json(
            session,
            f"leagues-classic/{OVERALL_LEAGUE_ID}/standings/",
            params={"page_standings": page},
        )
        standings = payload.get("standings", {})
        for manager in standings.get("results", []):
            records.append(
                {
                    "season": SEASON,
                    "snapshot_utc": snapshot_utc,
                    "stratum": stratum,
                    "source_page": page,
                    "manager_id": int(manager["entry"]),
                    "manager_name": manager.get("player_name"),
                    "team_name": manager.get("entry_name"),
                    "rank": manager.get("rank"),
                    "last_rank": manager.get("last_rank"),
                    "total_points": manager.get("total"),
                    "gameweek_points": manager.get("event_total"),
                }
            )
        time.sleep(REQUEST_DELAY_SECONDS)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(
            "The FPL standings API returned no managers for the selected pages."
        )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(STANDINGS_FILE, index=False)
    print(f"Saved {len(frame):,} managers to {STANDINGS_FILE}")
    return frame


# ---------------------------------------------------------------------------
# Manager gameweek history
# ---------------------------------------------------------------------------

def load_cached_histories() -> dict[int, dict]:
    """Load completed manager-history responses from the JSONL cache."""
    cached: dict[int, dict] = {}
    if not HISTORY_CACHE_FILE.exists():
        return cached

    with HISTORY_CACHE_FILE.open("r", encoding="utf-8") as cache_file:
        for line_number, line in enumerate(cache_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                cached[int(item["manager_id"])] = item["history"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                print(f"Skipping invalid history cache line {line_number}: {exc}")
    return cached


def collect_manager_histories(
    session: requests.Session,
    standings: pd.DataFrame,
) -> pd.DataFrame:
    """Fetch histories for the configured subset and save raw and tidy data."""
    manager_ids = standings["manager_id"].drop_duplicates().astype(int).tolist()
    if HISTORY_MANAGER_LIMIT is not None:
        manager_ids = manager_ids[:HISTORY_MANAGER_LIMIT]

    histories = load_cached_histories()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing_ids = [manager_id for manager_id in manager_ids if manager_id not in histories]

    if missing_ids:
        with HISTORY_CACHE_FILE.open("a", encoding="utf-8") as cache_file:
            for manager_id in tqdm(missing_ids, desc="Manager histories"):
                history = get_json(session, f"entry/{manager_id}/history/")
                histories[manager_id] = history
                cache_file.write(
                    json.dumps({"manager_id": manager_id, "history": history}) + "\n"
                )
                cache_file.flush()
                time.sleep(REQUEST_DELAY_SECONDS)

    rank_by_manager = (
        standings.drop_duplicates("manager_id")
        .set_index("manager_id")["rank"]
        .to_dict()
    )
    rows: list[dict] = []
    for manager_id in manager_ids:
        history = histories[manager_id]
        for gameweek in history.get("current", []):
            rows.append(
                {
                    "season": SEASON,
                    "manager_id": manager_id,
                    "overall_rank_at_sample": rank_by_manager.get(manager_id),
                    **gameweek,
                }
            )

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise RuntimeError(
            "No current-season gameweek history was returned for the selected managers."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(HISTORY_FILE, index=False)
    print(f"Saved {len(frame):,} manager-gameweek rows to {HISTORY_FILE}")
    return frame


def main() -> None:
    session = build_session()
    try:
        standings = collect_standings_sample(session)
        histories = collect_manager_histories(session, standings)
        print(
            f"Done: {len(standings):,} managers sampled; "
            f"{len(histories):,} gameweek history rows collected."
        )
    finally:
        session.close()


if __name__ == "__main__":
    main()
