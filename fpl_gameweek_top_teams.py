"""Collect the top FPL overall-ranked managers and their picks for one GW.

Change TARGET_GAMEWEEK to choose the week. The collector fetches only that
week's entry picks; it does not fetch manager season histories.

The public Overall standings endpoint exposes the latest completed GW rank
and one previous rank, not a full historical leaderboard. An older rank
cohort cannot be reconstructed safely from the standings API alone.

Both the standings scan and picks download are checkpointed. The picks stage
requests one manager's GW-specific picks at a time.
"""

from __future__ import annotations

import json
import math
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

TARGET_GAMEWEEK = 10
# Inclusive overall-rank range after TARGET_GAMEWEEK.
# For ranks 10,001–20,000, set START_RANK = 10_001 and END_RANK = 20_000.
START_RANK = 1
END_RANK = 10_000
MANAGER_COUNT = END_RANK - START_RANK + 1
MANAGERS_PER_PAGE = 50
REQUEST_DELAY_SECONDS = 0.25
SCAN_CHECKPOINT_PAGES = 1_000
PICK_CHECKPOINT_MANAGERS = 1_000
SELECTED_SHARD_ROWS = 50_000

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / f"gameweek_{TARGET_GAMEWEEK}_ranks_{START_RANK}_{END_RANK}"
RANK_SCAN_DIR = DATA_DIR / "rank_scan_parts"
COHORT_DIR = DATA_DIR / "cohort_parts"
MANAGER_PARTS_DIR = DATA_DIR / "manager_parts"
PICK_PARTS_DIR = DATA_DIR / "pick_parts"
FAILED_IDS_FILE = DATA_DIR / "failed_manager_ids.jsonl"
RANK_SCAN_STATE_FILE = DATA_DIR / "rank_scan_state.json"
RANK_SCAN_MANIFEST_FILE = DATA_DIR / "rank_scan_complete.json"
COHORT_MANIFEST_FILE = DATA_DIR / "cohort_complete.json"


# ---------------------------------------------------------------------------
# HTTP and JSON helpers
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
        {"User-Agent": "personal-fpl-analysis/0.1 (contact: mlg.wood1@gmail.com)"}
    )
    return session


def get_json(
    session: requests.Session,
    endpoint: str,
    params: dict | None = None,
) -> dict:
    url = f"{BASE_URL}{endpoint.lstrip('/')}"
    transient_errors = (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        json.JSONDecodeError,
    )
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=(10, 60))
            response.raise_for_status()
            return response.json()
        except requests.HTTPError:
            raise
        except transient_errors as exc:
            if attempt == 5:
                raise RuntimeError(
                    f"FPL request failed after six attempts: {endpoint}"
                ) from exc
            wait_seconds = min(2**attempt, 30)
            tqdm.write(
                f"Temporary FPL response failure for {endpoint}; "
                f"retrying in {wait_seconds}s ({attempt + 1}/5)."
            )
            time.sleep(wait_seconds)
    raise RuntimeError("FPL request retry loop exited unexpectedly.")


def write_json_atomic(path: Path, value: dict) -> None:
    temp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    temp_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp_path.replace(path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Gameweek and rank-snapshot helpers
# ---------------------------------------------------------------------------

def get_context(session: requests.Session) -> tuple[dict, dict[int, dict], dict]:
    """Read current FPL status, target GW and player metadata."""
    bootstrap = get_json(session, "bootstrap-static/")
    events = bootstrap.get("events", [])
    target_matches = [event for event in events if int(event["id"]) == TARGET_GAMEWEEK]
    if not target_matches:
        raise RuntimeError(f"Gameweek {TARGET_GAMEWEEK} was not found in bootstrap data.")
    target_event = target_matches[0]
    latest_finished = max(
        (int(event["id"]) for event in events if event.get("finished")),
        default=None,
    )
    next_event = min(
        (event for event in events if int(event["id"]) > TARGET_GAMEWEEK and not event.get("finished")),
        key=lambda event: int(event["id"]),
        default=None,
    )
    next_deadline = None
    if next_event and next_event.get("deadline_time"):
        next_deadline = datetime.fromisoformat(
            next_event["deadline_time"].replace("Z", "+00:00")
        )

    if not target_event.get("finished"):
        raise RuntimeError(
            f"GW{TARGET_GAMEWEEK} is not marked finished by FPL. Wait until it is "
            "complete before collecting its final ranks and picks."
        )

    teams = {
        int(team["id"]): team.get("name")
        for team in bootstrap.get("teams", [])
    }
    players = {
        int(player["id"]): {
            "player_name": player.get("web_name"),
            "element_type": player.get("element_type"),
            "team": teams.get(int(player["team"])),
        }
        for player in bootstrap.get("elements", [])
    }
    context = {
        "latest_finished": latest_finished,
        "next_deadline": next_deadline,
        "total_players": int(bootstrap.get("total_players", 0)),
    }
    return target_event, players, context


def get_rank_plan(context: dict) -> dict:
    """Select which rank field still represents the requested gameweek."""
    latest_finished = context["latest_finished"]
    if latest_finished == TARGET_GAMEWEEK:
        rank_field = "rank"
        scan_all_pages = False
    elif latest_finished == TARGET_GAMEWEEK + 1:
        rank_field = "last_rank"
        scan_all_pages = True
    else:
        raise RuntimeError(
            f"GW{TARGET_GAMEWEEK} is not the latest or immediately previous completed "
            f"gameweek (latest completed: {latest_finished}). The FPL standings API "
            "does not expose older overall-rank snapshots."
        )

    return {
        "rank_field": rank_field,
        "scan_all_pages": scan_all_pages,
        "latest_finished": latest_finished,
    }


def validate_rank_plan(session: requests.Session, expected_plan: dict) -> None:
    """Check periodically that a long standings scan still uses the same rank."""
    bootstrap = get_json(session, "bootstrap-static/")
    latest_finished = max(
        (
            int(event["id"])
            for event in bootstrap.get("events", [])
            if event.get("finished")
        ),
        default=None,
    )
    if latest_finished != expected_plan["latest_finished"]:
        raise RuntimeError(
            "The latest completed gameweek changed during the rank scan. The last "
            "complete checkpoint is retained, but the cohort snapshot cannot be "
            "completed safely from the new standings."
        )


def rank_scan_complete() -> bool:
    manifest = read_json(RANK_SCAN_MANIFEST_FILE)
    return bool(
        manifest
        and manifest.get("season") == SEASON
        and manifest.get("gameweek") == TARGET_GAMEWEEK
        and manifest.get("rank_field") in {"rank", "last_rank"}
    )


def cohort_is_complete() -> bool:
    manifest = read_json(COHORT_MANIFEST_FILE)
    return bool(
        manifest
        and manifest.get("season") == SEASON
        and manifest.get("gameweek") == TARGET_GAMEWEEK
        and manifest.get("start_rank") == START_RANK
        and manifest.get("end_rank") == END_RANK
        and manifest.get("manager_count") == MANAGER_COUNT
    )


def save_rank_scan_checkpoint(batch_number: int, rows: list[dict]) -> None:
    if not rows:
        return
    path = RANK_SCAN_DIR / f"rank_scan_{batch_number:05d}.parquet"
    temp = path.with_name(path.stem + ".tmp.parquet")
    pd.DataFrame.from_records(rows).to_parquet(temp, index=False)
    temp.replace(path)


def enrich_rank_scan_parts(session: requests.Session, plan: dict) -> None:
    """Ensure checkpoints include standings metadata, upgrading older shards."""
    required_columns = {"rank", "last_rank", "total_points", "standings_event_points"}
    parts = sorted(RANK_SCAN_DIR.glob("rank_scan_*.parquet"))
    for part_path in parts:
        frame = pd.read_parquet(part_path)
        if required_columns.issubset(frame.columns):
            continue

        manager_ids = set(frame["manager_id"].astype(int))
        part_number = int(part_path.stem.removeprefix("rank_scan_"))
        first_page = part_number * SCAN_CHECKPOINT_PAGES + 1
        pages_in_part = math.ceil(len(frame) / MANAGERS_PER_PAGE)
        metadata: dict[int, dict] = {}
        print(
            f"Adding standings metadata to {part_path.name} by rereading "
            f"{pages_in_part:,} saved standings pages."
        )
        for page_offset in tqdm(
            range(pages_in_part), desc=f"Enrich {part_path.stem}"
        ):
            page = first_page + page_offset
            payload = get_json(
                session,
                f"leagues-classic/{OVERALL_LEAGUE_ID}/standings/",
                params={"page_standings": page},
            )
            for row in payload.get("standings", {}).get("results", []):
                manager_id = int(row["entry"])
                if manager_id in manager_ids:
                    metadata[manager_id] = {
                        "last_rank": row.get("last_rank"),
                        "total_points": row.get("total"),
                        "standings_event_points": row.get("event_total"),
                    }
            if (page_offset + 1) % 100 == 0:
                validate_rank_plan(session, plan)
            time.sleep(REQUEST_DELAY_SECONDS)

        missing_ids = manager_ids.difference(metadata)
        if missing_ids:
            raise RuntimeError(
                f"Could not re-read standings metadata for {len(missing_ids):,} "
                f"managers in {part_path.name}; the checkpoint was left unchanged."
            )

        metadata_frame = pd.DataFrame(
            [metadata[int(manager_id)] for manager_id in frame["manager_id"]]
        )
        for column in required_columns:
            if column == "rank":
                frame[column] = frame["overall_rank_after_gameweek"]
            elif column in metadata_frame:
                frame[column] = metadata_frame[column].to_numpy()
        temp_path = part_path.with_name(part_path.stem + ".tmp.parquet")
        frame.to_parquet(temp_path, index=False)
        temp_path.replace(part_path)


def capture_rank_scan(
    session: requests.Session,
    plan: dict,
    snapshot_utc: str,
    total_players: int,
) -> None:
    """Page standings and checkpoint manager IDs with the target GW rank."""
    RANK_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    parts = sorted(RANK_SCAN_DIR.glob("rank_scan_*.parquet"))
    saved_count = sum(
        len(pd.read_parquet(path, columns=["manager_id"])) for path in parts
    )
    page = saved_count // MANAGERS_PER_PAGE + 1
    batch_number = len(parts)
    pages_since_checkpoint = 0
    checkpoint_rows: list[dict] = []
    page_total = (
        math.ceil(total_players / MANAGERS_PER_PAGE)
        if plan["scan_all_pages"]
        else math.ceil(END_RANK / MANAGERS_PER_PAGE)
    )
    page_progress = tqdm(
        total=page_total,
        initial=page - 1,
        desc=f"Scan GW{TARGET_GAMEWEEK} {plan['rank_field']} ranks",
    )
    candidate_count = saved_count
    scan_reached_end = not plan["scan_all_pages"] and candidate_count >= END_RANK

    try:
        while plan["scan_all_pages"] or candidate_count < END_RANK:
            payload = get_json(
                session,
                f"leagues-classic/{OVERALL_LEAGUE_ID}/standings/",
                params={"page_standings": page},
            )
            standings = payload.get("standings", {})
            results = standings.get("results", [])
            if not results:
                if plan["scan_all_pages"]:
                    scan_reached_end = True
                    break
                raise RuntimeError(
                    f"Standings ended at page {page}; only {candidate_count:,} "
                    f"of {END_RANK:,} ranks were available."
                )

            for row in results:
                if not plan["scan_all_pages"] and candidate_count >= END_RANK:
                    break
                candidate_count += 1
                checkpoint_rows.append(
                    {
                        "manager_id": int(row["entry"]),
                        "rank": row.get(plan["rank_field"]),
                        "overall_rank_after_gameweek": row.get(plan["rank_field"]),
                        "last_rank": row.get("last_rank"),
                        "total_points": row.get("total"),
                        "standings_event_points": row.get("event_total"),
                    }
                )

            page_progress.update(1)
            pages_since_checkpoint += 1
            if (
                pages_since_checkpoint >= SCAN_CHECKPOINT_PAGES
                or (not plan["scan_all_pages"] and candidate_count >= END_RANK)
            ):
                save_rank_scan_checkpoint(batch_number, checkpoint_rows)
                batch_number += 1
                checkpoint_rows.clear()
                pages_since_checkpoint = 0

            # The final page already supplied the requested cohort. Avoid an
            # unnecessary bootstrap recheck that could fail after all ranks
            # have been saved.
            if not plan["scan_all_pages"] and candidate_count >= END_RANK:
                break

            if not standings.get("has_next", len(results) == MANAGERS_PER_PAGE):
                scan_reached_end = True
                break
            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)

            if pages_since_checkpoint == 0 and page % SCAN_CHECKPOINT_PAGES == 1:
                validate_rank_plan(session, plan)
    finally:
        page_progress.close()

    if checkpoint_rows:
        save_rank_scan_checkpoint(batch_number, checkpoint_rows)

    if not plan["scan_all_pages"] and candidate_count < END_RANK:
        raise RuntimeError(
            f"Only {candidate_count:,} managers were found; "
            f"rank {END_RANK:,} was requested."
        )
    if plan["scan_all_pages"] and not scan_reached_end:
        raise RuntimeError("The all-manager standings scan did not reach its final page.")

    enrich_rank_scan_parts(session, plan)

    manifest = {
        "season": SEASON,
        "gameweek": TARGET_GAMEWEEK,
        "rank_field": plan["rank_field"],
        "scan_all_pages": plan["scan_all_pages"],
        "candidate_count": candidate_count,
        "snapshot_utc": snapshot_utc,
        "latest_finished_at_scan": plan["latest_finished"],
    }
    write_json_atomic(RANK_SCAN_MANIFEST_FILE, manifest)


def build_selected_cohort() -> None:
    """Choose the configured inclusive rank range from the completed scan."""
    manifest = read_json(RANK_SCAN_MANIFEST_FILE)
    if not manifest:
        raise RuntimeError("The standings rank scan has no completion manifest.")

    frames = [
        pd.read_parquet(
            path,
            columns=[
                "manager_id",
                "rank",
                "overall_rank_after_gameweek",
                "last_rank",
                "total_points",
                "standings_event_points",
            ],
        )
        for path in sorted(RANK_SCAN_DIR.glob("rank_scan_*.parquet"))
    ]
    if not frames:
        raise RuntimeError("No standings rank scan Parquet shards were found.")

    candidates = pd.concat(frames, ignore_index=True)
    candidates["overall_rank_after_gameweek"] = pd.to_numeric(
        candidates["overall_rank_after_gameweek"], errors="coerce"
    )
    candidates = candidates.dropna(subset=["overall_rank_after_gameweek"])
    if len(candidates) < END_RANK:
        raise RuntimeError(
            f"Only {len(candidates):,} managers have a GW{TARGET_GAMEWEEK} rank; "
            f"At least rank {END_RANK:,} is required."
        )

    selected = (
        candidates.sort_values(
            ["overall_rank_after_gameweek", "manager_id"], kind="mergesort"
        )
        .iloc[START_RANK - 1 : END_RANK]
        .reset_index(drop=True)
    )
    selected.insert(0, "snapshot_order", range(START_RANK, END_RANK + 1))
    selected.insert(0, "gameweek", TARGET_GAMEWEEK)
    selected.insert(0, "season", SEASON)

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    for part_number, start in enumerate(range(0, len(selected), SELECTED_SHARD_ROWS)):
        chunk = selected.iloc[start : start + SELECTED_SHARD_ROWS]
        path = COHORT_DIR / f"cohort_{part_number:05d}.parquet"
        temp = path.with_name(path.stem + ".tmp.parquet")
        chunk.to_parquet(temp, index=False)
        temp.replace(path)

    cohort_manifest = {
        "season": SEASON,
        "gameweek": TARGET_GAMEWEEK,
        "manager_count": len(selected),
        "start_rank": START_RANK,
        "end_rank": END_RANK,
        "rank_field": manifest["rank_field"],
        "rank_snapshot_utc": manifest["snapshot_utc"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rank_basis": "overall rank after target gameweek",
    }
    write_json_atomic(COHORT_MANIFEST_FILE, cohort_manifest)
    print(f"Saved {len(selected):,} managers in {COHORT_DIR}")


# ---------------------------------------------------------------------------
# GW-only picks and manager summaries
# ---------------------------------------------------------------------------

def load_completed_manager_ids() -> set[int]:
    completed: set[int] = set()
    if MANAGER_PARTS_DIR.exists():
        for path in MANAGER_PARTS_DIR.glob("manager_*.parquet"):
            ids = pd.read_parquet(path, columns=["manager_id"])["manager_id"]
            completed.update(ids.astype(int).tolist())
    if FAILED_IDS_FILE.exists():
        with FAILED_IDS_FILE.open("r", encoding="utf-8") as failed_file:
            for line in failed_file:
                if line.strip():
                    try:
                        completed.add(int(json.loads(line)["manager_id"]))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        continue
    return completed


def make_manager_record(
    cohort_row: dict,
    entry_history: dict,
    rank_snapshot_utc: str,
) -> dict:
    record = {
        "season": SEASON,
        "gameweek": TARGET_GAMEWEEK,
        "rank_snapshot_utc": rank_snapshot_utc,
        "manager_id": int(cohort_row["manager_id"]),
        "rank": cohort_row.get("rank", cohort_row["overall_rank_after_gameweek"]),
        "last_rank": cohort_row.get("last_rank"),
        "total_points": entry_history.get("total_points"),
        "gameweek_points": entry_history.get("points"),
        "gameweek_rank": entry_history.get("rank"),
        "overall_rank": entry_history.get("overall_rank"),
    }
    for source, target in (
        ("bank", "bank"),
        ("value", "team_value"),
        ("event_transfers", "gameweek_transfers"),
        ("event_transfers_cost", "gameweek_transfer_cost"),
        ("points_on_bench", "points_on_bench"),
    ):
        record[target] = entry_history.get(source)
    return record


def write_pick_checkpoint(
    batch_number: int,
    manager_rows: list[dict],
    pick_rows: list[dict],
) -> None:
    """Write picks first and manager summaries last as the resume marker."""
    manager_path = MANAGER_PARTS_DIR / f"manager_{batch_number:05d}.parquet"
    picks_path = PICK_PARTS_DIR / f"picks_{batch_number:05d}.parquet"
    manager_tmp = manager_path.with_name(manager_path.stem + ".tmp.parquet")
    picks_tmp = picks_path.with_name(picks_path.stem + ".tmp.parquet")
    pd.DataFrame.from_records(pick_rows).to_parquet(picks_tmp, index=False)
    pd.DataFrame.from_records(manager_rows).to_parquet(manager_tmp, index=False)
    picks_tmp.replace(picks_path)
    manager_tmp.replace(manager_path)


def next_pick_batch_number() -> int:
    numbers = [
        int(path.stem.removeprefix("manager_"))
        for path in MANAGER_PARTS_DIR.glob("manager_*.parquet")
        if path.stem.removeprefix("manager_").isdigit()
    ]
    return max(numbers, default=-1) + 1


def collect_picks(
    session: requests.Session,
    players: dict[int, dict],
) -> None:
    COHORT_DIR.mkdir(parents=True, exist_ok=True)
    MANAGER_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    PICK_PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add team names to pick shards created by earlier versions of this script.
    for pick_path in sorted(PICK_PARTS_DIR.glob("picks_*.parquet")):
        picks = pd.read_parquet(pick_path)
        if "team" in picks.columns:
            continue
        picks["team"] = picks["player_id"].map(
            lambda player_id: players.get(int(player_id), {}).get("team")
        )
        temp_path = pick_path.with_name(pick_path.stem + ".tmp.parquet")
        picks.to_parquet(temp_path, index=False)
        temp_path.replace(pick_path)

    completed_ids = load_completed_manager_ids()
    rank_snapshot_utc = read_json(COHORT_MANIFEST_FILE)["rank_snapshot_utc"]
    estimate_hours = MANAGER_COUNT * REQUEST_DELAY_SECONDS / 3600
    print(
        f"Fetching GW{TARGET_GAMEWEEK} picks for ranks "
        f"{START_RANK:,}–{END_RANK:,} ({MANAGER_COUNT:,} managers)."
    )
    print(
        f"At {REQUEST_DELAY_SECONDS:.2f}s between picks requests, the delay alone "
        f"takes at least {estimate_hours:,.1f} hours ({estimate_hours / 24:,.1f} days)."
    )
    print(f"Resuming with {len(completed_ids):,} already completed/failed manager IDs.")

    manager_batch: list[dict] = []
    picks_batch: list[dict] = []
    batch_number = next_pick_batch_number()
    failed_ids: set[int] = set()
    if FAILED_IDS_FILE.exists():
        with FAILED_IDS_FILE.open("r", encoding="utf-8") as failed_file:
            for line in failed_file:
                try:
                    failed_ids.add(int(json.loads(line)["manager_id"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass

    progress = tqdm(total=MANAGER_COUNT, desc=f"GW{TARGET_GAMEWEEK} picks")
    try:
        selected_count = 0
        for cohort_path in sorted(COHORT_DIR.glob("cohort_*.parquet")):
            cohort_rows = pd.read_parquet(cohort_path).to_dict(orient="records")
            for cohort_row in cohort_rows:
                selected_count += 1
                manager_id = int(cohort_row["manager_id"])
                if manager_id in completed_ids:
                    progress.update(1)
                    continue

                try:
                    response_data = get_json(
                        session,
                        f"entry/{manager_id}/event/{TARGET_GAMEWEEK}/picks/",
                    )
                except requests.HTTPError as exc:
                    if exc.response is None or exc.response.status_code != 404:
                        raise
                    with FAILED_IDS_FILE.open("a", encoding="utf-8") as failed_file:
                        failed_file.write(
                            json.dumps(
                                {
                                    "manager_id": manager_id,
                                    "gameweek": TARGET_GAMEWEEK,
                                    "status_code": 404,
                                }
                            )
                            + "\n"
                        )
                    completed_ids.add(manager_id)
                    failed_ids.add(manager_id)
                    progress.update(1)
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue

                pick_list = response_data.get("picks", [])
                if len(pick_list) != 15:
                    raise RuntimeError(
                        f"Manager {manager_id} returned {len(pick_list)} picks "
                        f"for GW{TARGET_GAMEWEEK}; expected 15."
                    )

                entry_history = response_data.get("entry_history", {})
                manager_batch.append(
                    make_manager_record(cohort_row, entry_history, rank_snapshot_utc)
                )
                for pick in pick_list:
                    player_id = int(pick["element"])
                    player = players.get(player_id, {})
                    picks_batch.append(
                        {
                            "season": SEASON,
                            "gameweek": TARGET_GAMEWEEK,
                            "manager_id": manager_id,
                            "rank": cohort_row.get(
                                "rank", cohort_row["overall_rank_after_gameweek"]
                            ),
                            "last_rank": cohort_row.get("last_rank"),
                            "total_points": entry_history.get("total_points"),
                            "gameweek_points": entry_history.get("points"),
                            "player_id": player_id,
                            "player_name": player.get("player_name"),
                            "element_type": player.get("element_type"),
                            "team": player.get("team"),
                            "squad_position": pick.get("position"),
                            "multiplier": pick.get("multiplier"),
                            "is_captain": pick.get("is_captain"),
                            "is_vice_captain": pick.get("is_vice_captain"),
                            "active_chip": response_data.get("active_chip"),
                        }
                    )

                completed_ids.add(manager_id)
                progress.update(1)
                time.sleep(REQUEST_DELAY_SECONDS)
                if len(manager_batch) >= PICK_CHECKPOINT_MANAGERS:
                    write_pick_checkpoint(batch_number, manager_batch, picks_batch)
                    batch_number += 1
                    manager_batch.clear()
                    picks_batch.clear()

        if selected_count != MANAGER_COUNT:
            raise RuntimeError(
                f"Expected {MANAGER_COUNT:,} selected managers; "
                f"found {selected_count:,} in the cohort dataset."
            )
    finally:
        progress.close()
        if manager_batch:
            write_pick_checkpoint(batch_number, manager_batch, picks_batch)

    successful = sum(
        len(pd.read_parquet(path, columns=["manager_id"]))
        for path in MANAGER_PARTS_DIR.glob("manager_*.parquet")
    )
    print(f"Manager summaries saved: {successful:,}")
    print(f"Unavailable GW{TARGET_GAMEWEEK} picks (HTTP 404): {len(failed_ids):,}")
    print(f"Manager summary dataset: {MANAGER_PARTS_DIR}")
    print(f"Player pick dataset: {PICK_PARTS_DIR}")


def collect() -> None:
    if START_RANK < 1 or END_RANK < START_RANK:
        raise ValueError("Set a valid inclusive rank range with 1 <= START_RANK <= END_RANK.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RANK_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    session = build_session()
    try:
        target_event, players, context = get_context(session)
        plan = get_rank_plan(context)
        print(
            f"GW{TARGET_GAMEWEEK} rank source: {plan['rank_field']}; "
            f"latest completed GW: {context['latest_finished']}."
        )

        if not rank_scan_complete():
            state = read_json(RANK_SCAN_STATE_FILE)
            if state:
                if (
                    state.get("season") != SEASON
                    or state.get("gameweek") != TARGET_GAMEWEEK
                    or state.get("rank_field") != plan["rank_field"]
                ):
                    raise RuntimeError(
                        "A partial rank scan exists for a different gameweek/rank basis. "
                        "Change the output folder or archive that partial scan before retrying."
                    )
                snapshot_utc = state["snapshot_utc"]
            else:
                snapshot_utc = datetime.now(timezone.utc).isoformat()
                write_json_atomic(
                    RANK_SCAN_STATE_FILE,
                    {
                        "season": SEASON,
                        "gameweek": TARGET_GAMEWEEK,
                        "rank_field": plan["rank_field"],
                        "snapshot_utc": snapshot_utc,
                    },
                )

            validate_rank_plan(session, plan)
            total_pages = math.ceil(context["total_players"] / MANAGERS_PER_PAGE)
            pages_to_scan = (
                total_pages
                if plan["scan_all_pages"]
                else math.ceil(END_RANK / MANAGERS_PER_PAGE)
            )
            delay_hours = pages_to_scan * REQUEST_DELAY_SECONDS / 3600
            print(f"Requested ranks: {START_RANK:,}–{END_RANK:,}")
            print(f"Estimated standings pages to scan: {pages_to_scan:,}")
            print(
                f"Standings request delays alone: {delay_hours:,.1f} hours, "
                "plus API response/retry time."
            )
            capture_rank_scan(session, plan, snapshot_utc, context["total_players"])

        if not cohort_is_complete():
            build_selected_cohort()
        else:
            print(f"Using saved GW{TARGET_GAMEWEEK} cohort: {COHORT_DIR}")

        collect_picks(session, players)
    finally:
        session.close()


if __name__ == "__main__":
    collect()
