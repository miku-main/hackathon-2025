# VALCoach - Valorant AI Pick Coach

VALCoach is a **second-screen assistant for Valorant esports** built for the **PrizePicks - Next-Gen Game Flow** hackathon.

It pulls **live player stats from vlrggapi**, builds simple **AI-style projections** for kills/assists, then ranks **Over / Under edges** on prop-like lines. Each pick comes with a **confidence label** and a **natural-language explanation** - so fans don't just see numbers, they also see *why*.

## What does it do?
- Connects to **[vlrggapi](https://github.com/axsddlr/vlrggapi)** (unofficial VLR.gg API) to pull Valorant pro stats.
- Converts **per-round stats -> per-map stats** (kills, assists).
- Builds a **projection** for kills/assists using rating + KAST (consistency).
- Synthesizes simple **PrizePicks-style lines** (e.g., `21.5 kills`).
- Computes an **edge score**: how far our projection is from the line (z-score-ish).
- Turns edge into:
    - `Lean Over` / `Lean Under` / `Stay Away`
    - `High` / `Medium` / `Low` confidence
    - Approximate **P(Over)** (probability of going over the line)
- Displays all of this in a **Streamlit dashboard**:
    - Region & time filters
    - Risk profile (Safe / Standard / YOLO)
    - Top 3 picks cards
    - Full ranked table
    - Per-player AI explanations

---

## How the model works
### 1. Data
From `vlrggapi /stats` (per region + timespan) we use:
- `kills_per_round`
- `assists_per_round`
- `rating`
- `kill_assists_survived_traded` (KAST %)
- (optionally) `rounds_played` if available

Build internal **player objects**
- `kills_per_map` = `kills_per_round x 22`
- `assists_per_map` = `assists_per_round x 22`
- `rating`, `kast`, `maps_played` (approx from rounds)
- `consistency` ∈ [0, 1] from rating + KAST

> 22 rounds per map is a **modeling assumption**: typical pro maps end around 13-9, 13-10, etc.
---
### 2. Projections
For each player and each stat (`kills`, `assists`), we compute a **projection**:
