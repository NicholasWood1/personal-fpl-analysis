"""Randomly sample FPL managers by sampling whole standings pages.

The rank interval is translated to standings page numbers (50 managers/page).
The script samples enough distinct pages to meet or exceed the requested
manager count, then keeps every manager returned on each selected page. This
means ranks on the interval's boundary pages can fall just outside the exact
rank range. It also fetches the requested gameweek's picks for every sampled
manager and saves them in a tidy Parquet file.

Example: sample 10,000 managers from ranks 100,000–1,000,000:
    python fpl_rank_page_sample.py --gameweek 9 --start-rank 100000 \
        --end-rank 1000000 --sample-size 10000 --seed 2026
"""

from __future__ import annotations

import argparse
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

BASE_URL = "https://fantasy.premierleague.com/api/"
OVERALL_LEAGUE_ID = 314
MANAGERS_PER_PAGE = 50


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "personal-fpl-analysis/0.1"})
    return session


def get_json(session: requests.Session, endpoint: str, params: dict | None = None) -> dict:
    response = session.get(
        f"{BASE_URL}{endpoint.lstrip('/')}", params=params, timeout=(10, 60)
    )
    response.raise_for_status()
    return response.json()


def write_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def save_page(output_dir: Path, page: int, rank_field: str, results: list[dict]) -> None:
    """Persist one page independently so reruns can resume safely."""
    rows = []
    for page_row, item in enumerate(results, start=1):
        rank = item.get(rank_field)
        rows.append({
            "standings_page": page,
            "page_row": page_row,
            "manager_id": int(item["entry"]),
            "rank": rank,
            "overall_rank_after_gameweek": rank,
            "last_rank": item.get("last_rank"),
            "total_points": item.get("total"),
            "standings_event_points": item.get("event_total"),
        })
    path = output_dir / f"page_{page:06d}.parquet"
    temp = path.with_name(path.stem + ".tmp.parquet")
    pd.DataFrame.from_records(rows).to_parquet(temp, index=False)
    temp.replace(path)


def fetch_manager_picks(
    session: requests.Session, manager_id: int, gameweek: int, path: Path
) -> dict | None:
    response = session.get(
        f"{BASE_URL}entry/{manager_id}/event/{gameweek}/picks/",
        timeout=(10, 60),
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    payload = response.json()
    temp = path.with_name(path.stem + ".tmp.json")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.replace(path)
    return payload


def build_picks_outputs(
    frame: pd.DataFrame, picks_dir: Path, output_dir: Path, gameweek: int
) -> tuple[int, int]:
    pick_rows: list[dict] = []
    manager_rows: list[dict] = []
    for manager in frame.to_dict(orient="records"):
        manager_id = int(manager["manager_id"])
        pick_path = picks_dir / f"manager_{manager_id}.json"
        if not pick_path.exists():
            continue
        payload = json.loads(pick_path.read_text(encoding="utf-8"))
        history = payload.get("entry_history") or {}
        manager_rows.append({
            "gameweek": gameweek, "manager_id": manager_id,
            "rank": manager.get("rank"),
            "overall_rank_after_gameweek": manager.get("overall_rank_after_gameweek"),
            "last_rank": manager.get("last_rank"),
            "total_points": manager.get("total_points"),
            "gameweek_points": history.get("points"),
            "gameweek_rank": history.get("rank"),
            "overall_rank": history.get("overall_rank"),
            "event_transfers": history.get("event_transfers"),
            "event_transfers_cost": history.get("event_transfers_cost"),
            "points_on_bench": history.get("points_on_bench"),
            "bank": history.get("bank"), "team_value": history.get("value"),
            "active_chip": payload.get("active_chip"),
            "automatic_subs": json.dumps(payload.get("automatic_subs", [])),
        })
        for pick in payload.get("picks", []):
            pick_rows.append({
                "gameweek": gameweek, "manager_id": manager_id,
                "rank": manager.get("rank"), "player_id": pick.get("element"),
                "squad_position": pick.get("position"),
                "multiplier": pick.get("multiplier"),
                "is_captain": pick.get("is_captain"),
                "is_vice_captain": pick.get("is_vice_captain"),
                "active_chip": payload.get("active_chip"),
            })

    unavailable_path = output_dir / "unavailable_picks.jsonl"
    unavailable_ids: set[int] = set()
    if unavailable_path.exists():
        for line in unavailable_path.read_text(encoding="utf-8").splitlines():
            try:
                unavailable_ids.add(int(json.loads(line)["manager_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    for filename, records in (
        ("manager_gameweek_summary.parquet", manager_rows),
        ("gameweek_picks.parquet", pick_rows),
    ):
        temp = output_dir / (filename.removesuffix(".parquet") + ".tmp.parquet")
        pd.DataFrame.from_records(records).to_parquet(temp, index=False)
        temp.replace(output_dir / filename)
    return len(manager_rows), len(unavailable_ids)


def run(args: argparse.Namespace) -> None:
    if args.start_rank < 1 or args.end_rank < args.start_rank:
        raise ValueError("Ranks must satisfy 1 <= start-rank <= end-rank.")
    if args.sample_size < 1:
        raise ValueError("sample-size must be positive.")

    first_page = (args.start_rank - 1) // MANAGERS_PER_PAGE + 1
    last_page = (args.end_rank - 1) // MANAGERS_PER_PAGE + 1
    eligible_pages = list(range(first_page, last_page + 1))
    pages_needed = math.ceil(args.sample_size / MANAGERS_PER_PAGE)
    if pages_needed > len(eligible_pages):
        raise ValueError(
            f"The rank band contains only {len(eligible_pages) * MANAGERS_PER_PAGE:,} "
            "page slots, fewer than the requested sample size."
        )
    selected_pages = sorted(random.Random(args.seed).sample(eligible_pages, pages_needed))

    output_dir = args.output_dir.resolve()
    pages_dir = output_dir / "sampled_page_parts"
    pages_dir.mkdir(parents=True, exist_ok=True)
    picks_dir = output_dir / "pick_parts"
    picks_dir.mkdir(parents=True, exist_ok=True)
    selected_pages_path = output_dir / "selected_pages.json"
    state_path = output_dir / "sample_state.json"
    manifest_path = output_dir / "sample_complete.json"
    page_manifest = {
        "season": args.season,
        "gameweek": args.gameweek,
        "start_rank": args.start_rank,
        "end_rank": args.end_rank,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "selected_pages": selected_pages,
    }
    if selected_pages_path.exists():
        old = json.loads(selected_pages_path.read_text(encoding="utf-8"))
        if old != page_manifest:
            raise RuntimeError(
                f"{output_dir} already contains a sample for different options. "
                "Choose another --output-dir or use the original options to resume."
            )
    else:
        write_json_atomic(selected_pages_path, page_manifest)

    session = build_session()
    try:
        bootstrap = get_json(session, "bootstrap-static/")
        events = bootstrap.get("events", [])
        target = next((event for event in events if int(event["id"]) == args.gameweek), None)
        if target is None or not target.get("finished"):
            raise RuntimeError(f"GW{args.gameweek} must exist and be marked finished.")
        latest = max((int(event["id"]) for event in events if event.get("finished")), default=None)
        if latest == args.gameweek:
            rank_field = "rank"
        elif latest == args.gameweek + 1:
            rank_field = "last_rank"
        else:
            raise RuntimeError(
                "FPL standings only retain current and previous rank snapshots; "
                "choose the latest or immediately previous finished gameweek."
            )

        snapshot_utc = datetime.now(timezone.utc).isoformat()
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("rank_field") != rank_field or state.get("latest_finished") != latest:
                raise RuntimeError("The rank basis changed since this sample started.")
            snapshot_utc = state["snapshot_utc"]
        else:
            write_json_atomic(state_path, {
                "season": args.season, "gameweek": args.gameweek,
                "rank_field": rank_field, "latest_finished": latest,
                "snapshot_utc": snapshot_utc,
            })

        existing_pages = {
            int(path.stem.removeprefix("page_"))
            for path in pages_dir.glob("page_*.parquet")
        }
        pending_pages = [page for page in selected_pages if page not in existing_pages]
        print(f"Rank field: {rank_field}; page size: {MANAGERS_PER_PAGE}.")
        print(
            f"Rank band spans pages {first_page:,}–{last_page:,}; sampling "
            f"{pages_needed:,} pages ({len(pending_pages):,} still to fetch)."
        )
        print(f"At {args.delay:g}s between requests, delay floor: {len(pending_pages) * args.delay / 60:.1f} minutes.")
        for index, page in enumerate(tqdm(pending_pages, desc="Fetch sampled pages")):
            payload = get_json(
                session,
                f"leagues-classic/{OVERALL_LEAGUE_ID}/standings/",
                {"page_standings": page},
            )
            results = payload.get("standings", {}).get("results", [])
            if not results:
                raise RuntimeError(f"Standings page {page} returned no managers.")
            save_page(pages_dir, page, rank_field, results)
            if index + 1 < len(pending_pages):
                time.sleep(args.delay)

        frame = pd.concat(
            [pd.read_parquet(p) for p in sorted(pages_dir.glob("page_*.parquet"))],
            ignore_index=True,
        )
        # Retain every manager from each selected page, including page-edge
        # managers just outside the requested rank interval.
        frame = frame.sort_values(["standings_page", "page_row"], kind="mergesort").reset_index(drop=True)
        temp = output_dir / "rank_page_sample.tmp.parquet"
        frame.to_parquet(temp, index=False)
        temp.replace(output_dir / "rank_page_sample.parquet")

        completed_pick_ids = {
            int(path.stem.removeprefix("manager_"))
            for path in picks_dir.glob("manager_*.json")
        }
        unavailable_path = output_dir / "unavailable_picks.jsonl"
        unavailable_ids: set[int] = set()
        if unavailable_path.exists():
            for line in unavailable_path.read_text(encoding="utf-8").splitlines():
                try:
                    unavailable_ids.add(int(json.loads(line)["manager_id"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        pending = [
            int(manager_id) for manager_id in frame["manager_id"]
            if int(manager_id) not in completed_pick_ids | unavailable_ids
        ]
        print(f"Fetching GW{args.gameweek} picks for {len(pending):,} managers.")
        for index, manager_id in enumerate(tqdm(pending, desc="Fetch gameweek picks")):
            try:
                payload = fetch_manager_picks(
                    session, manager_id, args.gameweek,
                    picks_dir / f"manager_{manager_id}.json",
                )
            except requests.RequestException as exc:
                # Keep the manager pending so a rerun can retry this request.
                raise RuntimeError(f"GW{args.gameweek} picks request failed for manager {manager_id}: {exc}") from exc
            if payload is None:
                with unavailable_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"manager_id": manager_id, "reason": "not_found"}) + "\n")
            if index + 1 < len(pending):
                time.sleep(args.delay)

        managers_with_picks, unavailable_count = build_picks_outputs(
            frame, picks_dir, output_dir, args.gameweek
        )

        write_json_atomic(manifest_path, {
            **page_manifest,
            "rank_field": rank_field,
            "latest_finished_at_scan": latest,
            "rank_snapshot_utc": snapshot_utc,
            "pages_fetched": len(selected_pages),
            "manager_rows_saved": len(frame),
            "managers_with_picks": managers_with_picks,
            "picks_unavailable_404": unavailable_count,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "rank_basis": "overall rank after target gameweek",
            "output_file": "rank_page_sample.parquet",
            "picks_file": "gameweek_picks.parquet",
            "manager_summary_file": "manager_gameweek_summary.parquet",
        })
        print(f"Saved {len(frame):,} managers from {len(selected_pages):,} pages to {output_dir}")
        print(f"GW{args.gameweek} picks saved for {managers_with_picks:,} managers; unavailable: {unavailable_count:,}.")
        print("Note: whole-page sampling can include managers just outside the exact rank band on its boundary pages.")
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--start-rank", type=int, required=True)
    parser.add_argument("--end-rank", type=int, required=True)
    parser.add_argument("--sample-size", type=int, required=True, help="Target number of managers; whole pages may exceed it.")
    parser.add_argument("--seed", type=int, default=2026, help="Set a different seed for another reproducible sample.")
    parser.add_argument("--delay", type=float, default=0.25, help="Pause between page requests; default 0.25 seconds.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("delay must be nonnegative")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "data" / f"gameweek_{args.gameweek}_page_sample_ranks_{args.start_rank}_{args.end_rank}_n_{args.sample_size}_seed_{args.seed}"
    return args


if __name__ == "__main__":
    run(parse_args())
