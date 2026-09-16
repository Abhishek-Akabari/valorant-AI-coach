"""
Henrik API client for the VALORANT AI coaching system.

Fetches a player's recent match data and computes the performance metrics
used by the analysis engine.
"""

import os
import requests
from collections import Counter

BASE = "https://api.henrikdev.xyz"
TRADE_WINDOW_MS = 3000      # a death is 'traded' if avenged within 3 seconds


def _headers():
    key = os.getenv("HENRIK_API_KEY")
    if not key:
        raise RuntimeError("HENRIK_API_KEY not found in environment")
    return {"Authorization": key}


def get_account(name, tag):
    """Look up a player's account. Returns the data dict, or None if not found."""
    r = requests.get(f"{BASE}/valorant/v1/account/{name}/{tag}",
                     headers=_headers(), timeout=20)
    if r.status_code != 200:
        return None
    return r.json().get("data")


def get_matches(name, tag, region, size=5, mode="competitive"):
    """Fetch recent matches. Returns a list (empty if none found)."""
    r = requests.get(
        f"{BASE}/valorant/v4/matches/{region}/pc/{name}/{tag}",
        headers=_headers(),
        params={"mode": mode, "size": size},
        timeout=30,
    )
    if r.status_code != 200:
        return []
    return r.json().get("data", [])


def _round_events(match, puuid):
    """Per-round facts about one player: first kill, first death, KAST."""
    kills = match.get("kills", [])
    n_rounds = len(match.get("rounds", []))

    # group kill events by round
    by_round = {}
    for k in kills:
        by_round.setdefault(k["round"], []).append(k)

    first_kills = first_deaths = kast_rounds = 0

    for rnd in range(n_rounds):
        events = sorted(by_round.get(rnd, []), key=lambda e: e["time_in_round_in_ms"])
        if not events:
            continue

        # first blood of the round
        opening = events[0]
        if opening["killer"]["puuid"] == puuid:
            first_kills += 1
        if opening["victim"]["puuid"] == puuid:
            first_deaths += 1

        # KAST: did the player get a Kill, Assist, get Traded, or Survive?
        got_kill = any(e["killer"]["puuid"] == puuid for e in events)
        got_assist = any(
            any(a["puuid"] == puuid for a in e.get("assistants", []))
            for e in events
        )
        deaths = [e for e in events if e["victim"]["puuid"] == puuid]
        survived = len(deaths) == 0

        traded = False
        for death in deaths:
            killer_id = death["killer"]["puuid"]
            t = death["time_in_round_in_ms"]
            # was the killer themselves killed shortly afterwards?
            traded = any(
                e["victim"]["puuid"] == killer_id
                and t < e["time_in_round_in_ms"] <= t + TRADE_WINDOW_MS
                for e in events
            )
            if traded:
                break

        if got_kill or got_assist or survived or traded:
            kast_rounds += 1

    return first_kills, first_deaths, kast_rounds, n_rounds


def get_player_stats(name, tag, region, size=5):
    """
    Fetch a player's recent matches and return metrics matching the
    analysis engine's expected keys, plus their most-played agent.
    """
    account = get_account(name, tag)
    if not account:
        return None, "Account not found. Please check the Riot ID and try again."

    puuid = account["puuid"]
    matches = get_matches(name, tag, region, size=size)
    if not matches:
        return None, "No recent competitive matches found for this account."

    totals = Counter()
    agents = Counter()
    matches_used = 0

    for match in matches:
        me = next((p for p in match["players"] if p["puuid"] == puuid), None)
        if not me:
            continue

        s = me["stats"]
        fk, fd, kast, n_rounds = _round_events(match, puuid)
        if n_rounds == 0:
            continue

        totals["rounds"]      += n_rounds
        totals["score"]       += s["score"]
        totals["kills"]       += s["kills"]
        totals["deaths"]      += s["deaths"]
        totals["assists"]     += s["assists"]
        totals["headshots"]   += s["headshots"]
        totals["bodyshots"]   += s["bodyshots"]
        totals["legshots"]    += s["legshots"]
        totals["damage"]      += s["damage"]["dealt"]
        totals["first_kills"] += fk
        totals["first_deaths"]+= fd
        totals["kast_rounds"] += kast

        agents[me["agent"]["name"].lower()] += 1
        matches_used += 1

    if totals["rounds"] == 0:
        return None, "Could not compute statistics from the available matches."

    r = totals["rounds"]
    shots = totals["headshots"] + totals["bodyshots"] + totals["legshots"]

    stats = {
        "Average Combat Score":           round(totals["score"] / r, 1),
        "Headshot %":                     round(totals["headshots"] / shots * 100, 1) if shots else None,
        "Kills Per Round":                round(totals["kills"] / r, 2),
        "Assists Per Round":              round(totals["assists"] / r, 2),
        "Kills:Deaths":                   round(totals["kills"] / totals["deaths"], 2) if totals["deaths"] else None,
        "Average Damage Per Round":       round(totals["damage"] / r, 1),
        "First Kills Per Round":          round(totals["first_kills"] / r, 3),
        "First Deaths Per Round":         round(totals["first_deaths"] / r, 3),
        "Kill, Assist, Trade, Survive %": round(totals["kast_rounds"] / r * 100, 1),
    }

    meta = {
        "name": f"{account['name']}#{account['tag']}",
        "main_agent": agents.most_common(1)[0][0] if agents else None,
        "matches_analysed": matches_used,
        "rounds_analysed": r,
    }
    return {"stats": stats, "meta": meta}, None