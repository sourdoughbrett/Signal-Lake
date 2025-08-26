# src/main.py
# streamlit run app/dashboard.py
"""
Runs:
  • live ingestion every 60 seconds
  • ensemble forecast every 5 minutes

Ctrl‑C to stop.
"""
from __future__ import annotations

import logging
import schedule
import time

from src.data_ingest import fetch_and_store
from src.db import init_schema
from src.models import generate_and_store_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ── clear any jobs that might already exist in this interpreter ───────────
schedule.clear()      # <─ NEW: prevents duplicate jobs on restart / reload
logging.debug("Cleared existing schedule jobs.")

# ------------------------------------------------------------------ 
# Job wrappers (catch & log exceptions so one failure won't kill loop)
# ------------------------------------------------------------------ 
def job_ingest() -> None:
    try:
        fetch_and_store()
    except Exception as e:
        logging.exception("Ingest job failed: %s", e)

def job_forecast() -> None:
    try:
        generate_and_store_forecast()
    except Exception as e:
        logging.exception("Forecast job failed: %s", e)

# ------------------------------------------------------------------ 
# Main
# ------------------------------------------------------------------ 
if __name__ == "__main__":
    init_schema()          # ensure tables exist

    # run both jobs once at startup so tables aren't empty
    job_ingest()
    job_forecast()

    # schedule recurring jobs
    schedule.every(60).seconds.do(job_ingest)
    schedule.every(5).minutes.do(job_forecast)

    logging.info(
        "Scheduler started — ingest every 60 s, forecast every 5 min "
        "(Ctrl‑C to stop)…"
    )
    while True:
        schedule.run_pending()
        time.sleep(1)

