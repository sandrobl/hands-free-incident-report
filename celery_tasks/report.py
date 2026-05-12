import logging
import time
from datetime import datetime, timedelta

from celery_app import celery_app
from db import get_sync_db, Report
from sqlalchemy import select
from geoalchemy2.functions import ST_DWithin
from sentence_transformers import SentenceTransformer, util


DUPLICATE_RADIUS_METERS = 10
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
    start_time = time.time()

    # 1. Fetch encrypted data from DB
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)

        duplicate_of = None
        duplicate_confidence = None
        best_score = 0.0
        best_id = None

        if report.location_upload is not None and report.description_full:
            cutoff = report.created_at - timedelta(days=DUPLICATE_LOOKBACK_DAYS)

            candidates = db.execute(
                select(Report).where(
                    Report.report_id != report_id,
                    Report.created_at >= cutoff,
                    Report.duplicate_of.is_(None),
                    ST_DWithin(
                        cast(Report.location_upload, Geography),
                        cast(report.location_upload, Geography),
                        DUPLICATE_RADIUS_METERS,
                    ),
                ),
            ).scalars().all()
            
            if candidates:
                model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder="/workspace/sentence_transformer", local_files_only=True)
                query_emb = model.encode(report.description_full, convert_to_tensor=True)

                
                for c in candidates:
                    if not c.description_full:
                        continue
                    cand_emb = model.encode(c.description_full, convert_to_tensor=True)
                    score = float(util.cos_sim(query_emb, cand_emb))
                    logging.info(f"Report {report_id} vs candidate {c.report_id} similarity score: {score}")
                    if score > best_score:
                        best_score = score
                        best_id = c.report_id
                        
                del model
            else:
                logging.info(f"Report {report_id} has no candidates for duplication (no nearby reports in the last {DUPLICATE_LOOKBACK_DAYS} days)")
        else:
            logging.info(f"Report {report_id} cannot be checked for duplicates (missing location or description)")
    
        if best_score >= SIMILARITY_THRESHOLD:
            duplicate_of = best_id
            duplicate_confidence = best_score
            report.status = "duplicate"
            logging.info(f"Report {report_id} marked as duplicate of {best_id} with confidence {best_score}")
        elif best_score >= CORRELATION_THRESHOLD:
            duplicate_of = best_id
            duplicate_confidence = best_score
            report.status = "correlation"
            logging.info(f"Report {report_id} correlated with {best_id} with score {best_score}")
        else:
            report.status = "report_generated"
            logging.info(f"Report {report_id} has no duplicates above threshold (best was {best_id} with score {best_score})")

  

        report.duplicate_of = duplicate_of
        report.duplicate_confidence = duplicate_confidence
        report.report_duration = time.time() - start_time

        db.commit()

    finally:
        db.close()
    logging.info(f"Finished report for {report_id}")