"""
engine/pick_engine.py

This module contains the "model" logic:
- Take cleaned player stats (from data.vlrgg_client).
- Compute projections for kills and assists.
- Synthesize prop-style lines.
- Compute edge scores and probabilities.
- Turn those into Over/Under/Stay Away recommendations + confidence.
- Generate a plain-English explanation per pick.
"""

import math
from dataclasses import dataclass
from typing import Literal, Dict, Any, List

from data.vlrgg_client import fetch_vlr_stats, build_players_from_stats

# Type alias for risk modes
RiskMode = Literal["safe", "standard", "yolo"]


@dataclass
class PickResult:
    """
    Represents one "pick" for a single player/stat combination.

    Example:
        - Player: TenZ
        - Stat: kills
        - Line: 21.5
        - Our projection: 23.0
        - Edge, probability, recommendation, etc.
    """
    player_id: str
    player_handle: str
    team_name: str
    role: str
    stat_type: str       # "kills" or "assists"
    line_value: float
    projected_value: float
    edge: float          # standardized edge score (rough z-score)
    probability_over: float
    recommendation: str  # "Lean Over", "Lean Under", "Stay Away"
    confidence: str      # "Low", "Medium", "High"
    explanation: str
    raw_player: Dict[str, Any]   # <— NEW


# ---------- Projection logic ----------


def _projection_for_stat(player: Dict[str, Any], stat_type: str) -> float:
    """
    Compute our projected per-map stat (kills or assists) for a player.

    We start from:
      - kills_per_map or assists_per_map (from data.vlrgg_client)
    Then apply small adjustments based on:
      - rating (how strong the player is overall)
      - KAST (consistency / impact)

    Math concept:
      projection = base + a*(rating - baseline_rating) + b*(KAST - baseline_kast)

    """
    base_kills = player["kills_per_map"]
    base_assists = player["assists_per_map"]
    rating = player["rating"]
    kast = player["kast"]

    # Baseline values that represent an "average" player
    baseline_rating = 1.0
    baseline_kast = 0.72

    # Deviations from the baseline
    dr = rating - baseline_rating        # how much better/worse than avg rating
    dk = kast - baseline_kast           # how much better/worse than avg KAST

    if stat_type == "kills":
        base = base_kills
        # Heavier weight on KAST and rating for kills
        proj = base + 4.0 * dr + 10.0 * dk
    else:  # assists
        base = base_assists
        # Assists might be slightly less sensitive to rating/KAST
        proj = base + 2.5 * dr + 6.0 * dk

    # Projection cannot be negative
    return max(0.0, proj)


def _line_for_stat(player: Dict[str, Any], stat_type: str) -> float:
    """
    Synthesize a PrizePicks-style line for a given stat.

    In a real integration, you'd pull actual lines from a provider.
    For the hackathon, we fake lines from the base per-map stats.

    Example:
      - If kills_per_map ≈ 21.3, we might set line_kills = 21.5
      - For assists, we keep lines smaller and non-zero.
    """
    if stat_type == "kills":
        base = player["kills_per_map"]
        # Round kills to the nearest integer, then add 0.5 to get a half-point line
        return round(base) + 0.5
    else:
        base = player["assists_per_map"]
        val = round(base)
        # Ensure at least 0.5 so we never show a 0.0 line
        return max(0.5, val - 0.5)


def _spread_for_stat(player: Dict[str, Any], stat_type: str) -> float:
    """
    Define the "spread" (uncertainty) in stat units for our edge calculation.

    Idea:
      edge ≈ (projection - line) / spread   (like a z-score)

    - Base spread is larger for kills than assists.
    - If the player is more "consistent", we slightly shrink the spread
      (we trust their mean more).
    """
    cons = player["consistency"]  # 0–1 consistency score

    if stat_type == "kills":
        base = 3.0   # typical kill variation
    else:
        base = 2.0   # assists are usually lower-variance

    # shrink spread by up to ~30% for highly consistent players
    adj = base * (1.0 - 0.3 * cons)

    # spread should never be too small
    return max(0.5, adj)


# ---------- Edge & probability ----------


def _edge_score(projection: float, line_value: float, spread: float) -> float:
    """
    Compute the edge as a rough "z-score":
        edge = (projection - line) / spread

    - Positive edge -> projection above the line (Over side).
    - Negative edge -> projection below the line (Under side).
    - Magnitude |edge| indicates strength.
    """
    if spread <= 0:
        spread = 0.5
    return (projection - line_value) / spread


def _edge_to_probability_over(edge: float) -> float:
    """
    Convert the edge (treated as a z-score) into a probability that
    the player goes Over the line, using the normal CDF.

    Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))

    We clip the result to [1%, 99%] to avoid extreme 0%/100% claims.
    """
    z = edge
    prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return max(0.01, min(0.99, prob))


# ---------- Recommendation & confidence ----------


def _recommendation_from_edge(edge: float, risk_mode: RiskMode, stat_type: str) -> str:
    """
    Turn an edge value into an Over/Under/Stay Away recommendation, based on:

      - risk_mode:
          "safe"     -> only strong edges become picks
          "standard" -> medium thresholds
          "yolo"     -> more aggressive, lower thresholds

      - stat_type:
          kills edges might require slightly higher thresholds than assists.
    """
    # Thresholds by risk mode and stat type
    if risk_mode == "safe":
        thr_kills, thr_assists = 1.0, 0.9
    elif risk_mode == "standard":
        thr_kills, thr_assists = 0.7, 0.6
    else:  # "yolo"
        thr_kills, thr_assists = 0.4, 0.35

    thr = thr_kills if stat_type == "kills" else thr_assists

    if edge >= thr:
        return "Lean Over"
    if edge <= -thr:
        return "Lean Under"
    return "Stay Away"


def _confidence_from_edge(edge: float) -> str:
    """
    Map the magnitude of edge to a qualitative confidence level.

    - High:  |edge| >= 1.5
    - Medium:0.8 <= |edge| < 1.5
    - Low:   |edge| < 0.8

    These cutoffs are arbitrary but easy to explain: bigger edge => more confident.
    """
    mag = abs(edge)
    if mag >= 1.5:
        return "High"
    if mag >= 0.8:
        return "Medium"
    return "Low"


# ---------- Explanation text ----------


def _explain_pick(
    player: Dict[str, Any],
    stat_type: str,
    line_value: float,
    projection: float,
    edge: float,
    probability_over: float,
    recommendation: str,
    risk_mode: RiskMode,
) -> str:
    """
    Create a human-readable explanation of why we like or dislike a line.

    It references:
      - per-map averages
      - maps played
      - rating & KAST
      - how far projection is from line
      - approximate probability of Over or Under
      - risk mode used (Safe/Standard/YOLO)
    """
    maps = player["maps_played"]
    rating = player["rating"]
    kast = player["kast"]
    base_val = player["kills_per_map"] if stat_type == "kills" else player["assists_per_map"]

    direction = "over" if projection > line_value else "under"
    risk_label = {"safe": "conservative", "standard": "balanced", "yolo": "aggressive"}[risk_mode]

    # Base description about averages and projection
    base = (
        f"{player['handle']} ({player['team']} {player['role']}) averages about "
        f"{base_val:.1f} {stat_type} per map over ~{maps} maps in this window. "
        f"Our projection for this slate is {projection:.1f} {stat_type}, "
        f"against a line of {line_value:.1f}, leaning {direction} by "
        f"{abs(projection - line_value):.1f}."
    )

    # Rating/KAST-based consistency context
    context = (
        f" Their rating is {rating:.2f} with KAST {kast:.2f}, "
        f"suggesting {'high consistency' if kast >= 0.78 else 'some volatility'}."
    )

    # Probability text, based on whether we lean Over or Under
    prob_over_pct = probability_over * 100.0
    if recommendation == "Lean Over":
        prob_text = (
            f" Interpreting the edge as a rough z-score, this corresponds to about "
            f"{prob_over_pct:.0f}% chance to go over the line."
        )
    elif recommendation == "Lean Under":
        prob_text = (
            f" Interpreting the edge as a rough z-score, this implies about "
            f"{(100 - prob_over_pct):.0f}% chance to stay under."
        )
    else:
        prob_text = (
            " The edge is small in either direction, so this looks close to a coin flip "
            "under our assumptions."
        )

    # Closing sentence tying it to the chosen risk profile
    rec_text = (
        f" Under a **{risk_label}** risk profile, we categorize this as "
        f"**{recommendation}**, with an edge of {edge:.2f} and that implied probability."
    )

    return base + context + prob_text + rec_text


# ---------- Public function: build_picks ----------


def build_picks(region: str, timespan: str, risk_mode: RiskMode) -> List[PickResult]:
    """
    Main entry point for the UI:

    1. Fetch stats via vlrggapi (through fetch_vlr_stats).
    2. Build clean player structures.
    3. For each player and each stat (kills, assists):
       - compute projection
       - synthesize line
       - compute spread, edge, and probability
       - derive recommendation + confidence
       - generate explanation
    4. Return a list of PickResult objects, sorted by absolute edge (biggest first).
    """
    # 1) Pull raw stats from vlrggapi and turn into a DataFrame
    df = fetch_vlr_stats(region=region, timespan=timespan)

    # 2) Convert DataFrame rows into structured player dictionaries
    players = build_players_from_stats(df)

    picks: List[PickResult] = []

    # 3) Iterate through each player and each stat type we care about
    for player in players:
        for stat_type in ("kills", "assists"):
            # Compute our projection
            projection = _projection_for_stat(player, stat_type)

            # Make up a PrizePicks-like line for this stat
            line_value = _line_for_stat(player, stat_type)

            # Estimate uncertainty
            spread = _spread_for_stat(player, stat_type)

            # Compute edge (z-score-ish)
            edge = _edge_score(projection, line_value, spread)

            # Convert edge to P(Over) using normal approximation
            prob_over = _edge_to_probability_over(edge)

            # Turn edge into Over/Under/Stay Away depending on risk mode
            rec = _recommendation_from_edge(edge, risk_mode, stat_type)

            # Determine High/Medium/Low confidence
            conf = _confidence_from_edge(edge)

            # Generate natural-language explanation
            explanation = _explain_pick(
                player=player,
                stat_type=stat_type,
                line_value=line_value,
                projection=projection,
                edge=edge,
                probability_over=prob_over,
                recommendation=rec,
                risk_mode=risk_mode,
            )

            # Build the PickResult dataclass instance
            picks.append(
                PickResult(
                    player_id=player["id"],
                    player_handle=player["handle"],
                    team_name=player["team"],
                    role=player["role"],
                    stat_type=stat_type,
                    line_value=line_value,
                    projected_value=projection,
                    edge=edge,
                    probability_over=prob_over,
                    recommendation=rec,
                    confidence=conf,
                    explanation=explanation,
                    raw_player=player,         # <— NEW: full stats dict to feed ChatGPT
                )
            )

    # 4) Sort picks by absolute edge (largest edges at the top)
    picks.sort(key=lambda p: abs(p.edge), reverse=True)
    return picks
