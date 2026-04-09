import gc
import logging
import os
import subprocess
import uuid
import torch


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

    try:
        # 2. Decrypt
        aes_key = crypto.decrypt_session_key(encrypted_session_key)

        with open(video_path, "rb") as f:
            encrypted_blob = bytearray(f.read())

        plaintext = bytearray(crypto.decrypt_video(aes_key, encrypted_blob))  

        # 3. Extract audio via /dev/shm (RAM-backed)
        shm_path = f"/dev/shm/{uuid.uuid4()}.mp4"
        try:
            with open(shm_path, "wb") as f:
                f.write(plaintext)
            
            # file is fully written and closed here
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", shm_path],
                capture_output=True, text=True
            )
            logging.info(f"ffprobe stdout: '{probe.stdout.strip()}' stderr: '{probe.stderr.strip()}'")
            if "audio" not in probe.stdout:
                raise RuntimeError(f"Video has no audio stream: {video_path}")

            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", shm_path, "-vn", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
                capture_output=True,
            )
            
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
        finally:
            if os.path.exists(shm_path):
                os.remove(shm_path)

        audio_np = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32).copy() / 32768.0
        
        if audio_np.size == 0:
            logging.error(f"Audio extraction failed for {report_id}: audio_np is empty. ffmpeg output size: {len(proc.stdout)}")
            raise RuntimeError("Audio extraction failed: audio_np is empty")
        
        if audio_np.size < 16000:
            raise RuntimeError(f"Audio too short for transcription: {audio_np.size} samples ({audio_np.size/16000:.2f}s), minimum is 1s")

        logging.info(f"audio_np shape: {audio_np.shape}, dtype: {audio_np.dtype}, min: {audio_np.min():.4f}, max: {audio_np.max():.4f}")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = whisper.load_model("large-v3", download_root="/workspace/whisper", device="cpu")
        logging.info(f"Whisper model device: {next(model.parameters()).device}")

        WHISPER_SAMPLE_RATE = 16000
        WHISPER_CHUNK = 30 * WHISPER_SAMPLE_RATE  # 480000 samples

        if audio_np.size < WHISPER_CHUNK:
            audio_np = np.pad(audio_np, (0, WHISPER_CHUNK - audio_np.size), mode='constant')

        result = model.transcribe(audio_np, fp16=False)
        
        logging.info(f"Whisper transcription for {report_id}: {result['text']}")
        
         # Update DB with transcription result
        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            report.description_full = result['text']
            db.commit()
        finally:
            db.close()

    except Exception as e:
        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            if "no audio stream" in str(e).lower():
                report.status = "Failed: no audio stream"
            elif "audio too short" in str(e).lower():
                report.status = "Failed: audio too short"
            elif "audio extraction failed" in str(e).lower():
                report.status = "Failed: audio extraction failed"
            elif "ffmpeg" in str(e).lower():
                report.status = "Failed: ffmpeg error"
            else:
                report.status = "Failed during ingest"
            db.commit()
        finally:
            db.close()
        logging.error(f"Ingest failed for {report_id}: {e}")
        raise

    finally:
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        try:
            crypto.shred(aes_key)
        except NameError:
            pass
        try:
            crypto.shred(encrypted_blob)
        except NameError:
            pass
        try:
            crypto.shred(plaintext)
        except NameError:
            pass

    logging.info(f"Finished ingest for {report_id}")
