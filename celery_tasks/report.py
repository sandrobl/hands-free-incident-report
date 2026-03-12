import logging

from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report


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
        encrypted_session_key = report.encrypted_session_key
        video_path = report.video_path
    finally:
        db.close()

    # 2. Decrypt
    aes_key = crypto.decrypt_session_key(encrypted_session_key)
    encrypted_blob = open(video_path, "rb").read()
    plaintext = crypto.decrypt_video(aes_key, encrypted_blob)

    # 3. TODO: generate final report, save to DB
    import time
    time.sleep(5)

    # 4. Shred sensitive material
    crypto.shred(aes_key)
    crypto.shred(plaintext)

    logging.info(f"Finished report for {report_id}")