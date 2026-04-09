import logging
from datetime import datetime, timedelta

from celery_app import celery_app
from db import get_sync_db, Report
from sqlalchemy import select
from geoalchemy2.functions import ST_DWithin
from sentence_transformers import SentenceTransformer, util


DUPLICATE_RADIUS_METERS = 50
DUPLICATE_LOOKBACK_DAYS = 7
SIMILARITY_THRESHOLD = 0.8
CORRELATION_THRESHOLD = 0.5

@celery_app.task(
    name="app.tasks.report.process_upload",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    queue="analysis.report",
    track_started=True,
)
def process_upload(self, report_id: str, batch_id: str):
    logging.info(f"Report Processing {report_id} in batch {batch_id}")

    # 1. Fetch encrypted data from DB
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)

        duplicate_of = None
        duplicate_confidence = None

        if report.location_upload is not None and report.description_full:
            cutoff = datetime.utcnow() - timedelta(days=DUPLICATE_LOOKBACK_DAYS)

            candidates = db.execute(
                select(Report).where(
                    Report.report_id != report_id,
                    Report.created_at >= cutoff,
                    Report.duplicate_of.is_(None),
                    ST_DWithin(Report.location_upload, report.location_upload, DUPLICATE_RADIUS_METERS)
                ),
            ).scalars().all()

            if candidates:
                model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="/workspace/sentence_transformer", local_files_only=True)
                query_emb = model.encode(report.description_full, convert_to_tensor=True)

                best_score = 0.0
                best_id = None
                for c in candidates:
                    if not c.description_full:
                        continue
                    cand_emb = model.encode(c.description_full, convert_to_tensor=True)
                    score = float(util.cos_sim(query_emb, cand_emb))
                    logging.info(f"Report {report_id} vs candidate {c.report_id} similarity score: {score}")
                    if score > best_score:
                        best_score = score
                        best_id = c.report_id
                    
                    if best_score >= SIMILARITY_THRESHOLD:
                        duplicate_of = best_id
                        duplicate_confidence = best_score
                        logging.info(f"Report {report_id} marked as duplicate of {best_id} with confidence {best_score}")
                    else:
                        logging.info(f"Report {report_id} has no duplicates above threshold (best was {best_id} with score {best_score})")
                    
                    if best_score >= CORRELATION_THRESHOLD:
                        logging.info(f"Report {report_id} has a correlated report {best_id} with score {best_score} (below duplicate threshold)")
            else:
                logging.info(f"Report {report_id} has no candidates for duplication (no nearby reports in the last {DUPLICATE_LOOKBACK_DAYS} days)")
        else:
            logging.info(f"Report {report_id} cannot be checked for duplicates (missing location or description)")
    
        report.duplicate_of = duplicate_of
        report.duplicate_confidence = duplicate_confidence
        report.status = "duplicate" if duplicate_of else "report_generated"
        db.commit()

    finally:
        db.close()
    logging.info(f"Finished report for {report_id}")