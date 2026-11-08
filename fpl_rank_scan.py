"""Download an FPL overall standings rank snapshot and build a rank cohort.

This is the ladder-only stage used by ``fpl_gameweek_top_teams.py``. It makes
one standings request per 50 managers, checkpoints Parquet shards, and can be
resumed by rerunning with the same options. It does not fetch picks or manager
histories.

Examples:
    python fpl_rank_scan.py --gameweek 9 --start-rank 1 --end-rank 2000000
    python fpl_rank_scan.py --gameweek 9 --start-rank 1 --end-rank 2000000 \
        --delay 0.1 --output-dir data/gw9_top_2m_ladder
"""

from __future__ import annotations

import argparse
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


def write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def save_part(directory: Path, part_number: int, rows: list[dict]) -> None:
    if not rows:
        return
    path = directory / f"rank_scan_{part_number:05d}.parquet"
    temp = path.with_name(path.stem + ".tmp.parquet")
    pd.DataFrame.from_records(rows).to_parquet(temp, index=False)
    temp.replace(path)


def run(args: argparse.Namespace) -> None:
    if args.start_rank < 1 or args.end_rank < args.start_rank:
        raise ValueError("Ranks must satisfy 1 <= start-rank <= end-rank.")

    output_dir = args.output_dir.resolve()
    scan_dir = output_dir / "rank_scan_parts"
    cohort_dir = output_dir / "cohort_parts"
    scan_dir.mkdir(parents=True, exist_ok=True)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "rank_scan_complete.json"
    state_path = output_dir / "rank_scan_state.json"

    session = build_session()
    try:
        bootstrap = get_json(session, "bootstrap-static/")
        events = bootstrap.get("events", [])
        target = next((e for e in events if int(e["id"]) == args.gameweek), None)
        if not target or not target.get("finished"):
            raise RuntimeError(f"GW{args.gameweek} must exist and be marked finished.")
        latest = max((int(e["id"]) for e in events if e.get("finished")), default=None)
        if latest == args.gameweek:
            rank_field, scan_all = "rank", False
        elif latest == args.gameweek + 1:
            rank_field, scan_all = "last_rank", True
        else:
            raise RuntimeError(
                "FPL standings only retain current and previous rank snapshots; "
                "choose the latest or immediately previous finished gameweek."
            )

        existing = sorted(scan_dir.glob("rank_scan_*.parquet"))
        saved_count = sum(len(pd.read_parquet(p, columns=["manager_id"])) for p in existing)
        page = saved_count // MANAGERS_PER_PAGE + 1
        part_number = len(existing)
        rows: list[dict] = []
        pages_per_part = max(1, args.checkpoint_pages)
        total_players = int(bootstrap.get("total_players", 0))
        pages_to_scan = math.ceil(total_players / MANAGERS_PER_PAGE) if scan_all else math.ceil(args.end_rank / MANAGERS_PER_PAGE)
        if saved_count and (manifest_path.exists() or state_path.exists()):
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            if state and (state.get("gameweek") != args.gameweek or state.get("rank_field") != rank_field):
                raise RuntimeError("Existing partial scan is for a different rank snapshot.")
        snapshot_utc = datetime.now(timezone.utc).isoformat()
        write_json_atomic(state_path, {"season": args.season, "gameweek": args.gameweek, "rank_field": rank_field, "snapshot_utc": snapshot_utc})

        print(f"GW{args.gameweek} standings field: {rank_field}; page size: {MANAGERS_PER_PAGE}.")
        print(f"Estimated requests: up to {pages_to_scan:,}; delay floor: {pages_to_scan * args.delay / 3600:.2f} hours.")
        progress = tqdm(total=pages_to_scan, initial=page - 1, desc="Scan standings")
        pages_since_save = 0
        count = saved_count
        try:
            while scan_all or count < args.end_rank:
                payload = get_json(session, f"leagues-classic/{OVERALL_LEAGUE_ID}/standings/", {"page_standings": page})
                standings = payload.get("standings", {})
                results = standings.get("results", [])
                if not results:
                    break
                for row in results:
                    if not scan_all and count >= args.end_rank:
                        break
                    count += 1
                    rank = row.get(rank_field)
                    rows.append({
                        "manager_id": int(row["entry"]),
                        "rank": rank,
                        "overall_rank_after_gameweek": rank,
                        "last_rank": row.get("last_rank"),
                        "total_points": row.get("total"),
                        "standings_event_points": row.get("event_total"),
                    })
                progress.update(1)
                pages_since_save += 1
                if pages_since_save >= pages_per_part or (not scan_all and count >= args.end_rank):
                    save_part(scan_dir, part_number, rows)
                    part_number += 1
                    rows.clear()
                    pages_since_save = 0
                if not scan_all and count >= args.end_rank:
                    break
                if not standings.get("has_next", len(results) == MANAGERS_PER_PAGE):
                    break
                page += 1
                time.sleep(args.delay)
        finally:
            progress.close()
        save_part(scan_dir, part_number, rows)

        frames = [pd.read_parquet(p) for p in sorted(scan_dir.glob("rank_scan_*.parquet"))]
        if not frames:
            raise RuntimeError("No standings rows were returned.")
        ladder = pd.concat(frames, ignore_index=True)
        ladder["overall_rank_after_gameweek"] = pd.to_numeric(ladder["overall_rank_after_gameweek"], errors="coerce")
        ladder = ladder.dropna(subset=["overall_rank_after_gameweek"])
        ladder = ladder.sort_values(["overall_rank_after_gameweek", "manager_id"], kind="mergesort")
        if len(ladder) < args.end_rank:
            raise RuntimeError(f"Only {len(ladder):,} ranked managers are available; requested rank {args.end_rank:,}.")

        for p in cohort_dir.glob("cohort_*.parquet"):
            p.unlink()
        selected = ladder.iloc[args.start_rank - 1:args.end_rank].copy().reset_index(drop=True)
        selected.insert(0, "snapshot_order", range(args.start_rank, args.end_rank + 1))
        selected.insert(0, "gameweek", args.gameweek)
        selected.insert(0, "season", args.season)
        for i, start in enumerate(range(0, len(selected), args.shard_rows)):
            path = cohort_dir / f"cohort_{i:05d}.parquet"
            temp = path.with_name(path.stem + ".tmp.parquet")
            selected.iloc[start:start + args.shard_rows].to_parquet(temp, index=False)
            temp.replace(path)

        write_json_atomic(manifest_path, {
            "season": args.season, "gameweek": args.gameweek,
            "rank_field": rank_field, "scan_all_pages": scan_all,
            "candidate_count": count, "snapshot_utc": snapshot_utc,
            "latest_finished_at_scan": latest,
        })
        write_json_atomic(output_dir / "cohort_complete.json", {
            "season": args.season, "gameweek": args.gameweek,
            "manager_count": len(selected), "start_rank": args.start_rank,
            "end_rank": args.end_rank, "rank_field": rank_field,
            "rank_snapshot_utc": snapshot_utc,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "rank_basis": "overall rank after target gameweek",
        })
        print(f"Saved {len(ladder):,} ladder rows and ranks {args.start_rank:,}–{args.end_rank:,} to {output_dir}")
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--start-rank", type=int, required=True)
    parser.add_argument("--end-rank", type=int, required=True)
    parser.add_argument("--delay", type=float, default=0.25, help="Pause between pages; default 0.25 seconds.")
    parser.add_argument("--checkpoint-pages", type=int, default=1000)
    parser.add_argument("--shard-rows", type=int, default=50000)
    parser.add_argument("--output-dir", type=Path, help="Defaults to data/gameweek_<gw>_ranks_<start>_<end>.")
    args = parser.parse_args()
    if args.delay < 0 or args.checkpoint_pages < 1 or args.shard_rows < 1:
        parser.error("delay must be nonnegative; checkpoint and shard sizes must be positive")
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "data" / f"gameweek_{args.gameweek}_ranks_{args.start_rank}_{args.end_rank}"
    return args


if __name__ == "__main__":
    run(parse_args())
