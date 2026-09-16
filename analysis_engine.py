"""
Analysis engine for the VALORANT AI coaching system.

Loads pre-computed professional benchmarks and compares an individual
player's statistics against them.
"""

import numpy as np
import pandas as pd

RECENT_YEARS = [2024, 2025, 2026]
LOWER_IS_BETTER = {'First Deaths Per Round'}

METRIC_WINDOWS = {
    'Rating':                         'all',
    'Average Combat Score':           'all',
    'Kills:Deaths':                   'all',
    'Kills Per Round':                'all',
    'Assists Per Round':              'all',
    'Headshot %':                     'recent',
    'Average Damage Per Round':       'recent',
    'First Kills Per Round':          'recent',
    'First Deaths Per Round':         'recent',
    'Kill, Assist, Trade, Survive %': 'recent',
    'Clutch Success % (clean)':       'recent',
}


def load_clean_data(path="processed/vct_all_years_clean.parquet"):
    """Load the cleaned VCT dataset saved by the data preparation notebook."""
    df = pd.read_parquet(path)
    df['is_aggregate'] = df['is_aggregate'].astype(bool)
    return df


def build_distributions(df, min_rounds=20):
    base = df[(~df['is_aggregate']) & (df['Rounds Played'] >= min_rounds)]
    out = {}
    for metric, window in METRIC_WINDOWS.items():
        sub = base if window == 'all' else base[base['Year'].isin(RECENT_YEARS)]
        series = pd.to_numeric(sub[metric], errors='coerce').dropna()
        if len(series) >= 100:
            out[metric] = series.values
    return out


def build_agent_distributions(df, agent, min_rounds=20):
    base = df[(~df['is_aggregate']) & (df['Rounds Played'] >= min_rounds)
              & (df['Year'].isin(RECENT_YEARS)) & (df['Agent'] == agent)]
    out = {}
    for metric in METRIC_WINDOWS:
        series = pd.to_numeric(base[metric], errors='coerce').dropna()
        if len(series) >= 20:
            out[metric] = series.values
    return out


def percentile_of(value, distribution):
    return float((distribution < value).mean() * 100)


def classify(pct, lower_is_better=False):
    if lower_is_better:
        pct = 100 - pct
    if pct >= 75: return "strength"
    if pct >= 50: return "above median"
    if pct >= 25: return "below median"
    return "priority weakness"


def compare_player(player_stats, distributions, agent_distributions=None):
    rows = []
    for metric, value in player_stats.items():
        if metric not in distributions or value is None:
            continue
        dist = distributions[metric]
        lower_better = metric in LOWER_IS_BETTER
        pct = percentile_of(value, dist)

        row = {
            'metric': metric,
            'player': value,
            'pro_median': round(float(np.median(dist)), 2),
            'percentile': round(100 - pct if lower_better else pct, 1),
            'verdict': classify(pct, lower_better),
        }
        if agent_distributions and metric in agent_distributions:
            adist = agent_distributions[metric]
            apct = percentile_of(value, adist)
            row['agent_median'] = round(float(np.median(adist)), 2)
            row['agent_percentile'] = round(100 - apct if lower_better else apct, 1)
        rows.append(row)

    return pd.DataFrame(rows)
# Agent roles. Used to interpret metrics in the context of what the role requires.
AGENT_ROLES = {
    'jett': 'duelist', 'raze': 'duelist', 'reyna': 'duelist',
    'phoenix': 'duelist', 'neon': 'duelist', 'yoru': 'duelist',
    'iso': 'duelist', 'waylay': 'duelist',
    'omen': 'controller', 'brimstone': 'controller', 'viper': 'controller',
    'astra': 'controller', 'harbor': 'controller', 'clove': 'controller',
    'sova': 'initiator', 'breach': 'initiator', 'skye': 'initiator',
    'kayo': 'initiator', 'fade': 'initiator', 'gekko': 'initiator',
    'tejo': 'initiator',
    'cypher': 'sentinel', 'killjoy': 'sentinel', 'sage': 'sentinel',
    'chamber': 'sentinel', 'deadlock': 'sentinel', 'vyse': 'sentinel',
}


def detect_patterns(comparison_df, agent):
    """
    Identify patterns that arise from combinations of metrics, which individual
    percentile comparisons do not express.

    Returns (role, findings). Findings are plain-language statements passed to
    the language model, so that identifying the pattern is done in code rather
    than left to the model to infer.
    """
    role = AGENT_ROLES.get((agent or "").lower())
    if not role:
        return None, []

    pct = {}
    for _, r in comparison_df.iterrows():
        value = r.get('agent_percentile')
        if pd.notna(value):
            pct[r['metric']] = float(value)

    findings = []
    fk   = pct.get('First Kills Per Round')
    fd   = pct.get('First Deaths Per Round')
    hs   = pct.get('Headshot %')
    kd   = pct.get('Kills:Deaths')
    kast = pct.get('Kill, Assist, Trade, Survive %')
    adr  = pct.get('Average Damage Per Round')
    apr  = pct.get('Assists Per Round')

    if role == 'duelist':
        if fk is not None and fd is not None and fk < 25 and fd > 55:
            findings.append(
                "Takes very few opening duels for a duelist while also dying first "
                "less often than typical for the agent. This indicates passive play "
                "in a role that requires creating space, rather than a discipline "
                "problem. Advice to play more cautiously would be counterproductive."
            )
        if fk is not None and fd is not None and fk > 55 and fd < 25:
            findings.append(
                "Enters frequently and wins those duels at a rate above typical for "
                "the agent, which is the core strength of the role."
            )

    if role in ('controller', 'sentinel', 'initiator'):
        if fd is not None and fd < 25:
            findings.append(
                f"Dies first noticeably more often than typical for a {role}, a role "
                "that generally benefits from surviving to use utility later in the round."
            )

    if hs is not None and kd is not None and hs > 60 and kd < 30:
        findings.append(
            "Aim is above typical for the agent while overall trade efficiency is well "
            "below it, suggesting the issue lies in positioning and timing rather than "
            "mechanical skill."
        )

    if kast is not None and adr is not None and kast < 30 and adr > 55:
        findings.append(
            "Deals damage at a reasonable rate but contributes to fewer rounds than "
            "typical, suggesting damage is often not converted into round impact."
        )

    if role == 'initiator' and apr is not None and apr < 30:
        findings.append(
            "Records fewer assists than typical for an initiator, a role built around "
            "setting up teammates."
        )

    return role, findings