import logging
import gc

from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

import requests
import torch



@celery_app.task(
    name="app.tasks.voice_text.process_upload",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    queue="analysis.voice_text",
    track_started=True,
)
def process_upload(self, report_id: str, batch_id: str):
    logging.info(f"Voice Text Processing {report_id} in batch {batch_id}")

    # 1. Fetch encrypted data from DB
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        encrypted_session_key = report.encrypted_session_key
        video_path = report.video_path
        description_full = report.description_full
    finally:
        db.close()

    # Optimize the full description for the segmentation step of SAM3
    model_id = "google/gemma-3-4b-it"

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        cache_dir="/workspace/gemma",
        local_files_only=True
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, cache_dir="/workspace/gemma", local_files_only=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Extract physical objects from the German text. Translate them to English. Output a comma separated list of English nouns only. Do not write anything else."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "German sentence: Das rote Auto parkt vor dem großen Baum."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "car, tree"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "German sentence: Wir haben zwei Stühle und einen Tisch bestellt."}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "chair, table"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"German sentence: {description_full}"}]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        decoded = processor.tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
    
    logging.info(f"Finished voice_text for {report_id} with result: {decoded.strip()}")

       # Update DB with transcription result
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        report.description_short = decoded.strip()
        db.commit()
    finally:
        db.close()
    
    del model
    torch.cuda.empty_cache()
    gc.collect()