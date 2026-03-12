import gc
import logging
import os
import pathlib
import subprocess
import uuid

import numpy as np
import whisper

from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report


@celery_app.task(
    name="app.tasks.ingest.process_upload",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    queue="analysis.ingest",
    track_started=True,
)
def process_upload(self, report_id: str, batch_id: str):
    logging.info(f"Ingest Processing upload {report_id} in batch {batch_id}")

    # 1. Fetch encrypted data from DB
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        encrypted_session_key = report.encrypted_session_key
        video_path = report.video_path
    finally:
        db.close()

    logging.info(f"Fetched data for {report.report_id} from DB, video at {video_path}, session key present: {encrypted_session_key is not None}")

    # 2. Decrypt
    aes_key = crypto.decrypt_session_key(encrypted_session_key)
    encrypted_blob = open(video_path, "rb").read()
    plaintext = crypto.decrypt_video(aes_key, encrypted_blob)

    # 3. Extract audio via /dev/shm (RAM-backed)
    shm_path = f"/dev/shm/{uuid.uuid4()}.mp4"
    try:
        with open(shm_path, "wb") as f:
            f.write(plaintext)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", shm_path, "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    finally:
        if os.path.exists(shm_path):
            os.remove(shm_path)
    audio_np = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    model = whisper.load_model("large-v3", download_root="/workspace/whisper")
    result = model.transcribe(audio_np)
    logging.info(f"Whisper transcription for {report_id}: {result['text']}")

    # Update DB with transcription result
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        report.description_full = result['text']
        db.commit()
    finally:
        db.close()

    del model
    gc.collect()

    # 4. Shred sensitive material
    crypto.shred(aes_key)
    crypto.shred(plaintext)

    logging.info(f"Finished ingest for {report_id}")
