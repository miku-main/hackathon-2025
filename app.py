"""
app.py

This is the Streamlit frontend for VALCoach.

Responsibilities:
- Provide controls for region, timespan, risk, stat type, and confidence filters.
- Call engine.build_picks(...) to get pick data.
- Show:
    - Summary badges (region, window, players, risk)
    - Simple metrics (Over/Under/Stay Away counts, average edge)
    - Top 3 picks as feature cards
    - A full table of all picks
    - Detailed expandable explanations per player/stat
"""

import pandas as pd
import streamlit as st

from engine.pick_engine import build_picks, RiskMode


# Configure basic Streamlit page layout
st.set_page_config(
    page_title="VALCoach – Valorant AI Pick Coach",
    layout="wide",
)

# ---------- CSS for styling (cards, badges, chips) ----------

st.markdown(
    """
    <style>
    .val-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1.5rem;
        margin-bottom: 0.75rem;
    }
    .val-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .val-subtitle {
        color: #b3b3b3;
        font-size: 0.95rem;
        margin-top: 0.25rem;
    }
    .val-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.03);
    }
    .badge-strong {
        border-color: rgba(56,189,248,0.8);
        background: rgba(56,189,248,0.12);
    }
    .card {
        border-radius: 0.9rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255,255,255,0.06);
        background: rgba(15,15,20,0.95);
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }
    .card-soft {
        border-radius: 0.9rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(255,255,255,0.05);
        background: rgba(20,20,30,0.9);
    }
    .card-title {
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }
    .card-sub {
        font-size: 0.85rem;
        color: #b3b3b3;
        margin-bottom: 0.35rem;
    }
    .chip {
        display: inline-block;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        font-size: 0.75rem;
        border: 1px solid rgba(255,255,255,0.2);
        margin-right: 0.25rem;
    }
    .chip-over {
        border-color: rgba(34,197,94,0.8);
        background: rgba(34,197,94,0.12);
    }
    .chip-under {
        border-color: rgba(248,113,113,0.9);
        background: rgba(248,113,113,0.12);
    }
    .chip-stay {
        border-color: rgba(148,163,184,0.9);
        background: rgba(148,163,184,0.12);
    }
    .chip-conf {
        border-color: rgba(250,204,21,0.9);
        background: rgba(250,204,21,0.1);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------

st.markdown(
    """
    <div class="val-header">
      <div>
        <h1 class="val-title">🎮 VALCoach – Valorant AI Pick Coach</h1>
        <p class="val-subtitle">
          Second-screen assistant for Valorant esports: pulls live stats from vlrggapi,
          builds simple projections, ranks edges on prop-style lines, and explains Over/Under leans.
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar controls (filters) ----------

st.sidebar.header("Controls")

# Region dropdown (which region's stats to pull)
region = st.sidebar.selectbox(
    "Region",
    options=["na", "eu", "ap", "br", "latam", "kr", "cn"],
    index=0,
)

# Timespan (how far back we look in stats)
timespan_label_to_value = {
    "Last 30 days": "30",
    "Last 90 days": "90",
    "All time": "all",
}
timespan_label = st.sidebar.selectbox(
    "Time span",
    options=list(timespan_label_to_value.keys()),
    index=0,
)
timespan = timespan_label_to_value[timespan_label]

# Risk profile radio (Safe / Standard / YOLO)
risk_label = st.sidebar.radio(
    "Risk profile",
    options=["Safe", "Standard", "YOLO"],
    index=1,
)
risk_mode: RiskMode = {
    "Safe": "safe",
    "Standard": "standard",
    "YOLO": "yolo",
}[risk_label]

# Stat types filter (kills / assists)
stat_type_filter = st.sidebar.multiselect(
    "Stat types",
    options=["kills", "assists", "headshot"],
    default=["kills", "assists"],
)

search_term = st.sidebar.text_input("Search player", "")

def matches_search(p, term: str) -> bool:
    """
    Return True if the search term is empty, or if it appears
    in the player's handle (case-insensitive).
    """
    term = term.strip().lower()
    if not term:
        return True
    return term in p.player_handle.lower()

# Confidence filter (Low / Medium / High)
min_confidence = st.sidebar.selectbox(
    "Minimum confidence",
    options=["Low", "Medium", "High"],
    index=0,
)

# Mapping confidence labels to a simple numeric rank
confidence_rank = {"Low": 0, "Medium": 1, "High": 2}

st.sidebar.caption("Data source: vlrggapi (unofficial VLR stats API).")

# ---------- Fetch & build picks from the engine ----------

try:
    with st.spinner("Pulling stats and computing AI edges..."):
        # Call into the engine (this does API fetch + modeling)
        picks = build_picks(region=region, timespan=timespan, risk_mode=risk_mode)
except Exception as e:
    st.error(f"Error fetching stats from vlrggapi: {e}")
    st.stop()

# Filter picks based on UI controls
filtered_picks = [
    p
    for p in picks
    if p.stat_type in stat_type_filter
    and confidence_rank[p.confidence] >= confidence_rank[min_confidence]
    and matches_search(p, search_term)
]

# If nothing passes the filters, show a warning and stop rendering
if not filtered_picks:
    st.warning("No picks match your filters. Try lowering confidence or changing risk/region.")
    st.stop()

# ---------- Badges & top-level metrics ----------

total_players = len({p.player_handle for p in picks})
total_picks = len(picks)

# Badge strip summarizing context
st.markdown(
    f"""
    <div class="val-badges">
      <span class="badge badge-strong">Region: {region.upper()}</span>
      <span class="badge">Window: {timespan_label}</span>
      <span class="badge">Players (pool): {total_players}</span>
      <span class="badge">Total picks: {total_picks}</span>
      <span class="badge">Risk: {risk_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Simple metrics
over_count = sum(1 for p in filtered_picks if p.recommendation == "Lean Over")
under_count = sum(1 for p in filtered_picks if p.recommendation == "Lean Under")
stay_count = sum(1 for p in filtered_picks if p.recommendation == "Stay Away")
avg_edge = sum(abs(p.edge) for p in filtered_picks) / len(filtered_picks)

# Four metric cards in one row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Over leans</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{over_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Under leans</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{under_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Stay away</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{stay_count}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Avg edge (|z|)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{avg_edge:.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Top 3 picks section ----------

st.markdown("### 🔝 Top 3 AI Picks")

# Take the first 3 picks (already sorted by |edge| in the engine)
top3 = filtered_picks[:3]
cols_top = st.columns(len(top3))

for p, col in zip(top3, cols_top):
    # Determine chip style based on recommendation
    chip_class = (
        "chip-over"
        if p.recommendation == "Lean Over"
        else "chip-under"
        if p.recommendation == "Lean Under"
        else "chip-stay"
    )
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Player name
        st.markdown(f'<div class="card-title">{p.player_handle}</div>', unsafe_allow_html=True)
        # Team + role
        st.markdown(
            f'<div class="card-sub">{p.team_name} • {p.role or "Unknown role"}</div>',
            unsafe_allow_html=True,
        )
        # Recommendation chips (Over/Under/Stay + confidence)
        st.markdown(
            f"""
            <div style="margin-bottom: 0.45rem;">
              <span class="chip {chip_class}">{p.recommendation}</span>
              <span class="chip chip-conf">{p.confidence} conf</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Line, projection, edge, probability
        st.markdown(
            f"""
            <div style="font-size:0.85rem; color:#d1d5db;">
              Stat: <b>{p.stat_type.upper()}</b><br/>
              Line: <b>{p.line_value:.1f}</b> &nbsp; Projection: <b>{p.projected_value:.1f}</b><br/>
              Edge: <b>{p.edge:.2f}</b> &nbsp; P(Over): <b>{p.probability_over*100:.0f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tabs: full table + explanations ----------

tab_table, tab_explain = st.tabs(["📊 Full table", "🧠 AI explanations"])

with tab_table:
    st.markdown("#### Ranked pick list")

    # Convert PickResult objects into a DataFrame for display
    table_df = pd.DataFrame(
        [
            {
                "Player": p.player_handle,
                "Team": p.team_name,
                "Stat": p.stat_type,
                "Line": p.line_value,
                "Projection": round(p.projected_value, 2),
                "Edge": round(p.edge, 2),
                "P(Over)": round(p.probability_over * 100, 1),
                "Recommendation": p.recommendation,
                "Confidence": p.confidence,
            }
            for p in filtered_picks
        ]
    )

    # Display the table with Streamlit's dataframe component
    st.dataframe(table_df, use_container_width=True, hide_index=True)

with tab_explain:
    st.markdown("#### Individual AI breakdowns")

    # One expander per pick, with the full explanation text
    for p in filtered_picks:
        label = (
            f"{p.player_handle} – {p.stat_type.upper()} "
            f"({p.recommendation}, {p.confidence}, edge {p.edge:.2f})"
        )
        with st.expander(label):
            st.write(f"**Team:** {p.team_name}")
            st.write(f"**Role:** {p.role or 'Unknown'}")
            st.write(
                f"**Line:** {p.line_value:.1f} | "
                f"**Projection:** {p.projected_value:.1f} | "
                f"**Edge:** {p.edge:.2f} | "
                f"**P(Over):** {p.probability_over*100:.1f}%"
            )
            st.write(p.explanation)

# Footer disclaimer
st.caption(
    "Hackathon prototype for a PrizePicks-style track. Uses vlrggapi for stats. "
    "Not affiliated with Riot Games, VLR, or PrizePicks. Not betting advice."
)
