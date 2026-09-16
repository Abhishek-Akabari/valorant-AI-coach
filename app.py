"""
VALORANT AI Coaching System — web interface.

Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from analysis_engine import (
    load_clean_data,
    build_distributions,
    build_agent_distributions,
    compare_player,
    detect_patterns,
)
from coaching_engine import comparison_to_text, generate_coaching
from henrik_client import get_player_stats

load_dotenv()

QUESTIONNAIRE_URL = "https://forms.gle/P6f9MpX5QUGMmz3dA"

st.set_page_config(page_title="VALORANT AI Coach", page_icon="🎯")


# --------------------------------------------------------------------------
# Cached data loading: the benchmark data is large, so load it only once.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading professional benchmarks...")
def load_benchmarks():
    df = load_clean_data()
    return df, build_distributions(df)


# --------------------------------------------------------------------------
# Consent screen (required by the approved ethics application)
# --------------------------------------------------------------------------
def consent_screen():
    st.title("VALORANT AI Coaching System")
    st.caption("MSc dissertation research — Brunel University London")

    st.markdown(
        """
        ### About this study

        This tool analyses your recent VALORANT match statistics, compares them
        against professional VALORANT Champions Tour benchmarks, and generates
        personalised coaching feedback.

        It forms part of an MSc Artificial Intelligence Engineering dissertation
        at Brunel University London.

        **What taking part involves**

        You enter your own Riot ID below. The tool retrieves your publicly
        available match statistics, analyses them, and shows you a coaching
        report. You are then invited to complete a short anonymous questionnaire
        about the feedback, which takes about 10 to 15 minutes.

        **Your data**

        Your Riot ID is used only to retrieve your match statistics and is not
        stored by the researcher. The questionnaire is anonymous and collects no
        personal identifying information. Because responses cannot be linked back
        to you, they cannot be withdrawn once submitted.

        **Please note**

        VALORANT is a first-person shooter containing stylised, non-graphic
        depictions of violence typical of the genre. This study only involves
        reviewing your own statistics and does not require you to play any
        additional matches.

        Taking part is entirely voluntary and you may stop at any time.
        """
    )

    st.divider()
    st.markdown("### Consent")

    c1 = st.checkbox("I have read and understood the information above.")
    c2 = st.checkbox("I am over the age of 18.")
    c3 = st.checkbox(
        "I understand that my Riot ID is used only to retrieve my match data "
        "and is not stored by the researcher."
    )
    c4 = st.checkbox(
        "I understand that no personal identifying data is collected, and that "
        "I cannot withdraw my questionnaire responses once submitted."
    )
    c5 = st.checkbox("I agree to take part in this study.")

    all_agreed = all([c1, c2, c3, c4, c5])

    if st.button("Continue", type="primary", disabled=not all_agreed):
        st.session_state.consented = True
        st.rerun()

    if not all_agreed:
        st.caption("Please confirm all statements above to continue.")

    with st.expander("Contact details"):
        st.markdown(
            """
            **Researcher:** Abhishek Sanjaybhai Akabari — 2523424@brunel.ac.uk
            **Supervisor:** Dr. Stephen Swift, Department of Computer Science,
            Brunel University London
            """
        )


# --------------------------------------------------------------------------
# Main coaching interface
# --------------------------------------------------------------------------
def coaching_screen():
    st.title("Your coaching report")

    with st.form("riot_id_form"):
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            name = st.text_input("Riot ID name", placeholder="YourName")
        with col2:
            tag = st.text_input("Tag", placeholder="1234")
        with col3:
            region = st.selectbox("Region", ["eu", "na", "ap", "kr", "latam", "br"])

        matches = st.slider("Recent competitive matches to analyse", 3, 10, 5)
        submitted = st.form_submit_button("Get my coaching report", type="primary")

    if not submitted:
        st.info(
            "Enter your Riot ID above. This is the name and tag shown in game, "
            "for example PlayerName#1234."
        )
        return

    if not name or not tag:
        st.error("Please enter both your Riot ID name and tag.")
        return

    df, distributions = load_benchmarks()

    with st.spinner("Retrieving your match data..."):
        result, error = get_player_stats(name.strip(), tag.strip().lstrip("#"),
                                         region, size=matches)

    if error:
        st.error(error)
        return

    stats = result["stats"]
    meta = result["meta"]
    agent = meta["main_agent"]

    st.success(
        f"Analysed {meta['matches_analysed']} matches "
        f"({meta['rounds_analysed']} rounds). Most played agent: {agent.title()}."
    )

    with st.spinner("Comparing against professional benchmarks..."):
        agent_dist = build_agent_distributions(df, agent)
        comparison = compare_player(stats, distributions, agent_dist)
        role, findings = detect_patterns(comparison, agent)

    if comparison.empty:
        st.error("Could not compare your statistics against the benchmarks.")
        return

    with st.spinner("Generating your coaching feedback..."):
        player_text = comparison_to_text(
            comparison, agent=agent.title(), role=role, findings=findings
        )
        coaching = generate_coaching(player_text)

    st.markdown("### Coaching feedback")
    st.markdown(coaching)

    with st.expander("See the statistics behind this feedback"):
        st.caption(
            "Percentiles show where you sit relative to professional players. "
            "The agent columns compare you only against professionals playing the "
            "same agent, which is usually the more meaningful comparison. The "
            "coaching feedback above is based on the agent comparison."
        )
        display = comparison.rename(columns={
            "metric": "Metric",
            "player": "You",
            "pro_median": "Pro median",
            "percentile": "Percentile",
            "agent_median": f"{agent.title()} median",
            "agent_percentile": f"{agent.title()} percentile",
            "verdict": "Assessment",
        })

        column_order = ["Metric", "You", "Pro median", "Percentile",
                        f"{agent.title()} median", f"{agent.title()} percentile",
                        "Assessment"]
        display = display[[c for c in column_order if c in display.columns]]

        st.dataframe(display, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### Please share your feedback")
    st.markdown(
        "Now that you have read your coaching report, please complete the short "
        "anonymous questionnaire. Your responses are essential to this research."
    )
    st.link_button("Open the questionnaire", QUESTIONNAIRE_URL, type="primary")

    if st.button("Analyse a different account"):
        st.rerun()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    if "consented" not in st.session_state:
        st.session_state.consented = False

    if st.session_state.consented:
        coaching_screen()
    else:
        consent_screen()


if __name__ == "__main__":
    main()