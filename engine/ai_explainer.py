# engine/ai_explainer.py

from typing import Any, Dict, List
from openai import OpenAI
import streamlit as st

client = OpenAI()


SYSTEM_PROMPT = """
You are VALCoach, an esports betting assistant for Valorant.

You:
- Explain picks to a fan in clear, friendly language.
- Always ground your reasoning in the stats you are given
  (kills/assists per map, rating, KAST, consistency, edge, line, probability, etc.).
- Never invent stats or match history beyond what is provided.
- If you don't have some info, say so briefly instead of guessing.

Style:
- First, list 3–6 bullet points that highlight the most important numbers.
- Then write 1–2 short paragraphs explaining the recommendation
  (why Lean Over / Lean Under / Stay Away) in plain English.
- Be concise but specific; name the stats when you use them.
"""


def _format_player_context(player: Dict[str, Any]) -> str:
    """
    Turn the raw player dict into a readable block the model can use
    as 'retrieved knowledge'. This is your RAG-style context.
    """
    lines = []
    for k, v in sorted(player.items()):
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _build_initial_messages(pick: Any) -> List[Dict[str, str]]:
    """
    Build the base messages for explaining a single pick.
    Assumes pick is a PickResult from engine.pick_engine.
    """
    player = pick.raw_player
    context_block = _format_player_context(player)

    pick_summary = f"""
    Player: {pick.player_handle}
    Team: {pick.team_name}
    Role: {pick.role}
    Region (if present): {player.get('region', 'unknown')}
    Stat type: {pick.stat_type}
    Line: {pick.line_value:.1f}
    Projection: {pick.projected_value:.1f}
    Edge: {pick.edge:.2f}
    P(Over): {pick.probability_over * 100:.1f}%
    Recommendation: {pick.recommendation} ({pick.confidence} confidence)
    """.strip()

    user_content = f"""
    Here is all the structured context we have for this player and matchup.

    === PICK SUMMARY ===
    {pick_summary}

    === RAW PLAYER STATS ===
    {context_block}

    Using ONLY this information, explain to a Valorant fan why this is
    the recommendation. Follow the style rules from the system prompt.
    """

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content.strip()},
    ]


def generate_initial_explanation(pick: Any) -> str:
    """
    Call ChatGPT once to generate the main pick explanation.
    """
    messages = _build_initial_messages(pick)
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",  # or another model you have access to
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content


def answer_followup(pick: Any, history: List[Dict[str, str]]) -> str:
    """
    Answer a follow-up question about this pick.
    `history` is a list of {"role": "user"|"assistant", "content": "..."}.

    We reattach the same RAG context each time (so the model always sees
    the stats) and then append the running chat turns.
    """
    base_messages = _build_initial_messages(pick)

    system_msg = base_messages[0]
    context_msg = {
        "role": "assistant",
        "content": base_messages[1]["content"],
    }

    messages: List[Dict[str, str]] = [system_msg, context_msg]
    messages.extend(history)

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3,
    )
    return completion.choices[0].message.content
