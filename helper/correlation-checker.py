import logging
import time
from datetime import timedelta
import sys
from pathlib import Path

from geoalchemy2 import Geography

from sqlalchemy import select, text, cast
from geoalchemy2.functions import ST_DWithin
from sentence_transformers import SentenceTransformer, util

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from db import get_sync_db, Report

DUPLICATE_RADIUS_METERS = 10
DUPLICATE_LOOKBACK_DAYS = 7
MODEL_PATH              = "/workspace/sentence_transformer"
MODEL_NAME              = "paraphrase-multilingual-MiniLM-L12-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def check_report(report: Report, db, model: SentenceTransformer) -> dict:
    result = {
        "report_id":   report.report_id,
        "correlation": None,
        "correlation_with": None,
        "report_description_full": report.description_full,
        "correlation_description_full": None,
    }

    cutoff = report.created_at - timedelta(days=DUPLICATE_LOOKBACK_DAYS)

    candidates = db.execute(
        select(Report).where(
            Report.report_id != report.report_id,
            Report.created_at >= cutoff,
            Report.duplicate_of.is_(None),
            ST_DWithin(
                cast(Report.location_upload, Geography),
                cast(report.location_upload, Geography),
                DUPLICATE_RADIUS_METERS,
            ),
        )
    ).scalars().all()

    if not candidates:
        log.info(f"[{report.report_id}] No nearby candidates in the last {DUPLICATE_LOOKBACK_DAYS} days")
        return result

    query_emb  = model.encode(report.description_full, convert_to_tensor=True)
    best_score = 0.0
    best_id    = None
    best_desc  = None

    for c in candidates:
        if not c.description_full:
            continue
        cand_emb = model.encode(c.description_full, convert_to_tensor=True)
        score    = float(util.cos_sim(query_emb, cand_emb))
        log.info(f"[{report.report_id}] vs [{c.report_id}] → score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_id    = c.report_id
            best_desc  = c.description_full

    if best_id:
        result["correlation"] = best_score
        result["correlation_with"] = best_id
        result["correlation_description_full"] = best_desc
    log.info(f"[{report.report_id}] Best match: {best_id} @ {best_score:.4f}")
    return result


def main():
    db = get_sync_db()
    try:
        reports = db.execute(select(Report)).scalars().all()

        log.info(f"Loading model '{MODEL_NAME}'...")
        model   = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_PATH, local_files_only=True)
        total   = len(reports)
        t_start = time.time()

        for i, report in enumerate(reports, 1):
            log.info(f"Processing {i}/{total} — report {report.report_id}")
            result = check_report(report, db, model)
            db.execute(
                text("""
                    INSERT INTO report_correlation_results (
                        report_id,
                        correlation,
                        correlation_with,
                        report_description_full,
                        correlation_description_full
                    )
                    VALUES (
                        :report_id,
                        :correlation,
                        :correlation_with,
                        :report_description_full,
                        :correlation_description_full
                    )
                    ON CONFLICT (report_id)
                    DO UPDATE SET correlation = EXCLUDED.correlation,
                                  correlation_with = EXCLUDED.correlation_with,
                                  report_description_full = EXCLUDED.report_description_full,
                                  correlation_description_full = EXCLUDED.correlation_description_full
                """),
                result,
            )
            db.commit()

        del model
        log.info(f"Done. Processed {total} reports in {time.time() - t_start:.1f}s")

    finally:
        db.close()


if __name__ == "__main__":
    main()