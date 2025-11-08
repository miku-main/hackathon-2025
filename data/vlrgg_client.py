"""
data/vlrgg_client.py

This module is responsible for:
- Talking to the vlrggapi REST API to fetch Valorant player stats.
- Converting the raw JSON data into a cleaned, structured list of "player" dictionaries
  that the model/engine can consume.
"""

import re
from typing import Any, Dict, List

import pandas as pd
import requests

# Base URL for the vlrggapi project (from the GitHub README)
VLRGG_BASE_URL = "https://vlrggapi.vercel.app"

# Modeling assumption:
# - A typical Valorant pro map runs ~22 rounds (e.g., 13–9, 13–10, etc.).
# - We use this to convert kills_per_round -> kills_per_map.
EXPECTED_ROUNDS_PER_MAP = 22

AGENT_TO_ROLE: Dict[str,str] = {
    # Duelist
    "jett": "duelist",
    "neon": "duelist",
    "raze": "duelist",
    "yoru": "duelist",
    "iso": "duelist",
    "reyna": "duelist",
    "waylay": "duelist",
    "pheonix": "duelist",
    
    # Controller
    "omen": "controller",
    "clove": "controller",
    "viper": "controller",
    "brimstone": "controller",
    "harbor": "controller",
    "astra": "controller",
    
    # Initiator
    "gekko": "initiator",
    "sova": "initiator",
    "fade": "initiator",
    "kayo": "initiator",
    "breach": "initiator",
    "skye": "initiator",
    "tejo": "initiator",
    
    # Sentinels
    "cypher": "sentinels",
    "killjoy": "sentinels",
    "vyse": "sentinels",
    "deadlock": "sentinels",
    "sage": "sentinels",
    "chamber": "sentinels",
    "veto": "sentinels",
}

def _float_safe(x, default: float = 0.0) -> float:
    """
    Convert a value to float, but if anything goes wrong (None, empty string, bad format),
    return the given default.

    Example:
        _float_safe("1.23") -> 1.23
        _float_safe("not a number", 0.0) -> 0.0
    """
    try:
        return float(str(x))
    except Exception:
        return default


def _parse_kast(kast_raw: Any) -> float:
    """
    Parse KAST% (Kill / Assist / Survive / Trade) from a string like "72%" into 0.72.

    If parsing fails, we fall back to a reasonable baseline (0.70).
    """
    try:
        s = str(kast_raw).replace("%", "")  # "72%" -> "72"
        return float(s) / 100.0             # "72" -> 0.72
    except Exception:
        return 0.70
    
def _infer_role_from_agents(agents_raw: Any) -> str:
    """
    Infer a tactical role (Duelist / Controller / Initiator / Sentinel / Flex)
    based on the agents a player has been playing.

    We expect `agents_raw` to be either:
      - a string (e.g. "Jett, Raze, Phoenix")
      - a list of agent names (e.g. ["Jett", "Raze"])
      - or None / missing.

    For now, we:
      - normalize names to lowercase,
      - check against AGENT_TO_ROLE,
      - pick the first matching role we see,
      - default to "Flex" if we see multiple different roles,
      - default to "Unknown" if we get nothing useful.
    """
    if not agents_raw:
        return "Unknown"

    # Turn into a list of strings.
    if isinstance(agents_raw, str):
        # Split on commas or slashes, e.g. "Jett, Raze" or "Jett / Raze"
        parts = re.split(r"[,/]+", agents_raw)
        agent_names = [p.strip().lower() for p in parts if p.strip()]
    elif isinstance(agents_raw, (list, tuple)):
        agent_names = [str(a).strip().lower() for a in agents_raw if str(a).strip()]
    else:
        # Unknown format
        return "Unknown"

    if not agent_names:
        return "Unknown"

    # Collect all roles we can recognize
    roles_seen = []
    for name in agent_names:
        role = AGENT_TO_ROLE.get(name)
        if role and role not in roles_seen:
            roles_seen.append(role)

    if not roles_seen:
        # None of the agents matched our mapping
        return "Unknown"
    if len(roles_seen) == 1:
        # Clear single-role player: e.g. pure Duelist main
        return roles_seen[0]

    # Mixed roles (e.g., plays Duelist + Initiator); call that Flex.
    return "Flex"


def fetch_vlr_stats(region: str = "na", timespan: str = "30") -> pd.DataFrame:
    """
    Fetch player stats from vlrggapi for a specific region and time window.

    Args:
        region: Region code used by vlrggapi, e.g. "na", "eu", "ap".
        timespan: Time window (as a string), e.g. "30" (last 30 days), "90", or "all".

    Returns:
        A pandas DataFrame where each row is a player and each column is a stat,
        as provided by the vlrggapi /stats endpoint.
    """
    # Build query parameters for the API call
    params = {"region": region, "timespan": timespan}

    # Perform the HTTP GET request
    resp = requests.get(f"{VLRGG_BASE_URL}/stats", params=params, timeout=10)

    # Raise an error if the status code is not 200 OK
    resp.raise_for_status()

    # Parse the JSON payload
    payload = resp.json()

    # The relevant data is under payload["data"]["segments"]
    segments = payload.get("data", {}).get("segments", [])

    # If no segments returned, return an empty DataFrame
    if not segments:
        return pd.DataFrame()

    # Convert the list of dicts into a pandas DataFrame for easier processing
    return pd.DataFrame(segments)


def build_players_from_stats(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Transform vlrggapi /stats rows into a simple player structure used by the model.

    For each player row in the DataFrame, we produce a dictionary with fields:

        {
          "id": unique machine-friendly id,
          "handle": in-game name,
          "team": org tag (or "Unknown"),
          "role": (placeholder for now, "Unknown"),
          "kills_per_map": expected kills in a typical map,
          "assists_per_map": expected assists in a typical map,
          "rating": overall player rating,
          "kast": KAST value (0–1),
          "maps_played": approximate number of maps in this window,
          "consistency": a 0–1 score based on rating and KAST
        }

    This function is the bridge between the raw API and our modeling logic.
    """
    players: List[Dict[str, Any]] = []

    # If there's no data, return an empty list
    if df.empty:
        return players

    # Work on a copy so we don't modify the original DataFrame
    df = df.copy()

    # Convert "rating" into a numeric column for sorting players by performance
    df["rating_float"] = df["rating"].apply(lambda x: _float_safe(x, 1.0))

    # Keep the top ~40 players by rating to keep the UI manageable
    df = df.sort_values("rating_float", ascending=False).head(40)

    # Iterate over each row (each player)
    for _, row in df.iterrows():
        # Player handle (in-game name)
        handle = str(row["player"]).strip()

        # Team / org name; default to "Unknown" if missing
        team = str(row.get("org", "")).strip() or "Unknown"

        # Build a simple id from team + handle and clean it for safety
        raw_id = f"{team}_{handle}".lower()
        player_id = re.sub(r"[^a-z0-9]+", "", raw_id) or handle.lower()

        # Basic stats from the API
        rating = _float_safe(row["rating"], 1.0)
        kpr = _float_safe(row.get("kills_per_round", 0.8), 0.8)     # kills per round
        apr = _float_safe(row.get("assists_per_round", 0.3), 0.3)   # assists per round
        kast = _parse_kast(row.get("kill_assists_survived_traded", "70%"))

        # Convert from per-round --> per-map using our average rounds assumption
        kills_per_map = kpr * EXPECTED_ROUNDS_PER_MAP
        assists_per_map = apr * EXPECTED_ROUNDS_PER_MAP

        # Approximate total rounds played.
        # If the API doesn't give "rounds_played", assume ~10 maps worth of rounds.
        rounds_played = _float_safe(
            row.get("rounds_played", EXPECTED_ROUNDS_PER_MAP * 10),
            EXPECTED_ROUNDS_PER_MAP * 10,
        )
        # Convert total rounds into approximate maps
        maps_played = max(1, round(rounds_played / EXPECTED_ROUNDS_PER_MAP))

        # Build a simple consistency score from rating + KAST, scaled 0–1.
        # This is NOT rigorous math; it's a hacky but interpretable metric.
        # The idea:
        #   - If KAST is much higher than 0.65 and rating is much higher than 0.9,
        #     then consistency should approach 1.
        consistency = max(
            0.0,
            min(
                1.0,
                0.5 * (kast - 0.65) / 0.15 + 0.5 * (rating - 0.9) / 0.4,
            ),
        )
        
        agents_raw = (
            row.get("agents")
            or row.get("agent")
        )
        role = _infer_role_from_agents(agents_raw)

        # Finally, collect the data for this player
        players.append(
            {
                "id": player_id,
                "handle": handle,
                "team": team,
                "role": role,
                "kills_per_map": kills_per_map,
                "assists_per_map": assists_per_map,
                "rating": rating,
                "kast": kast,
                "maps_played": maps_played,
                "consistency": consistency,
            }
        )

    return players
