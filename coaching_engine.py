"""
Coaching engine for the VALORANT AI coaching system.

Converts the benchmark comparison into text and generates natural-language
coaching feedback using the Gemini API.
"""

import os
import time

from google import genai
from google.genai import types, errors

import logging
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

MODEL = "gemini-3.6-flash"
TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are an experienced VALORANT coach giving feedback to a player
based on their recent match data compared against professional benchmarks.

RULES
- Use ONLY the numbers provided. Never invent or estimate statistics.
- Percentiles compare this player against professional players, so low percentiles
  are normal for most players and are not a reason to be harsh.
- Where an agent percentile is given, prioritise it over the overall percentile.
  The agent's role determines what good performance looks like.
- Express percentiles as whole numbers, or in plain language such as "in the bottom
  10% of Jett players". Never write a decimal as an ordinal.
- Describe what the data shows. Do not claim that one metric causes another.
- Identify the TWO most important areas to work on, not everything.
- For each, give one specific, practical drill or habit they can act on.
- Do not repeat in the priorities a point already made in the summary.
- Be direct and encouraging. Do not be patronising.
- Where ANALYSIS FINDINGS are provided, treat them as established conclusions and
  build your advice around them. Do not contradict them.

FORMAT
1. One short paragraph summarising their overall performance.
2. "Priority 1:" the biggest issue, why it matters, and what to practise.
3. "Priority 2:" the second issue, same structure.
4. One sentence on something they are doing acceptably.

Keep the whole response under 250 words."""


def _client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=key)


def comparison_to_text(comparison_df, agent=None, role=None, findings=None):
    """Convert the comparison and any detected patterns into text for the LLM."""
    lines = []
    if agent:
        header = f"Main agent: {agent}"
        if role:
            header += f" (role: {role})"
        lines.append(header)
    lines.append("")
    lines.append("Metric | Player | Pro median | Percentile | "
                 "Agent median | Agent percentile | Verdict")

    for _, r in comparison_df.iterrows():
        agent_med = r.get('agent_median', 'n/a')
        agent_pct = r.get('agent_percentile', 'n/a')
        lines.append(
            f"{r['metric']} | {r['player']} | {r['pro_median']} | "
            f"{r['percentile']}th | {agent_med} | {agent_pct} | {r['verdict']}"
        )

    if findings:
        lines.append("")
        lines.append("ANALYSIS FINDINGS (identified from the data, use these):")
        for f in findings:
            lines.append(f"- {f}")

    return "\n".join(lines)

def generate_coaching(player_text, model=MODEL, temperature=TEMPERATURE,
                      max_retries=3):
    """Generate coaching feedback, retrying if the API is temporarily unavailable."""
    client = _client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=f"{SYSTEM_PROMPT}\n\nPLAYER DATA\n{player_text}",
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return response.text

        except errors.ServerError:
            wait = 2 ** attempt
            print(f"Server busy (attempt {attempt + 1}/{max_retries}), "
                  f"retrying in {wait}s...")
            time.sleep(wait)

        except errors.ClientError as e:
            return f"Could not generate coaching: {e}"

    return ("The coaching service is currently unavailable. "
            "Please try again in a few minutes.")