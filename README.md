# VALORANT AI Coaching System

MSc Artificial Intelligence dissertation, Brunel University London (CS5500).

A coaching tool that retrieves a player's recent VALORANT match statistics,
compares them against six years of professional VALORANT Champions Tour data,
and uses a large language model to generate natural-language coaching feedback.

**Live application:** https://valorant-ai-coach-4aoyvsrj63ywpmkgebje96.streamlit.app/

## Structure

- `analysis_engine.py` — benchmark construction, percentile comparison, role-based pattern detection
- `henrik_client.py` — retrieval and processing of player match data
- `coaching_engine.py` — prompt construction and language model integration
- `experiment_log.py` — logging of prompt versions and outputs
- `app.py` — web interface
- `VAL.ipynb` — data preparation and benchmark construction
- `LLM_INTEGRATION.ipynb` — prompt development
- `HENRIK_API.ipynb` — API integration and live testing
- `prompt_experiments/` — record of prompt versions and generated output

## Running locally

Requires a Gemini API key and a Henrik API key in a `.env` file:

GEMINI_API_KEY=...
HENRIK_API_KEY=...

Then: `pip install -r requirements.txt` and `streamlit run app.py`