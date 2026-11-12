"""Save FPL player IDs, names, positions and teams to a CSV file."""

from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://fantasy.premierleague.com/api/"
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "player_information.csv"


def fetch_player_information() -> pd.DataFrame:
    """Fetch the current player list and map position/team names."""
    response = requests.get(
        f"{BASE_URL}bootstrap-static/",
        headers={"User-Agent": "personal-fpl-analysis/0.1"},
        timeout=(10, 60),
    )
    response.raise_for_status()
    bootstrap = response.json()

    positions = {
        int(position["id"]): position["singular_name_short"]
        for position in bootstrap.get("element_types", [])
    }
    teams = {
        int(team["id"]): team["name"]
        for team in bootstrap.get("teams", [])
    }

    rows = [
        {
            "player_id": int(player["id"]),
            "player_name": player.get("web_name"),
            "position": positions.get(int(player["element_type"])),
            "team": teams.get(int(player["team"])),
        }
        for player in bootstrap.get("elements", [])
    ]
    return pd.DataFrame(rows, columns=["player_id", "player_name", "position", "team"])


def main() -> None:
    players = fetch_player_information()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(players):,} players to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
