import logging
import gc

from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report
# Gemma 3
# from transformers import AutoProcessor, Gemma3ForConditionalGeneration
# Gemma 4
from transformers import AutoProcessor, AutoModelForMultimodalLM

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
        description_full = report.description_full
    finally:
        db.close()

    # Optimize the full description for the segmentation step of SAM3
    #model_id = "google/gemma-3-4b-it"

    # Gemma 4
    model_id = "google/gemma-4-e2b-it"  # or gemma-4-e2b-it for smaller/faster
    cache_dir = "/workspace/gemma-4"  # Adjust cache directory as needed
    try:
        # Gemma 4
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            local_files_only=True
        ).eval()

        # model = Gemma3ForConditionalGeneration.from_pretrained(
        #     model_id, 
        #     device_map="auto", 
        #     torch_dtype=torch.bfloat16,
        #     cache_dir="/workspace/gemma",
        #     local_files_only=True
        # ).eval()

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": (
                    "Extract the most important physical object from the sentence. "
                    "Output EXACTLY 4 to 6 English noun phrases as a comma-separated list, most specific first. "
                    "Include color if mentioned. Translate to English if necessary. "
                    "Do not write anything else. Do not explain. Only output the comma-separated list."
                )}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Sentence: Das blaue Auto ist falsch parkiert."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "blue car, car, vehicle, automobile"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Sentence: The red car is parked in front of the big tree."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "red car, car, vehicle, automobile"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Sentence: As you can see there are a couple of things on my table like a pen, a mouse but the most important thing here is the video game console."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "gaming console, console, game controller, electronic device"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Sentence: Wir haben zwei Stühle und einen Tisch bestellt."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "chair, dining chair, seat, furniture, seating"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Sentence: There is a broken glass bottle on the sidewalk."}]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "broken glass bottle, glass bottle, bottle, broken glass, debris"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Sentence: {description_full}"}]
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
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
            logging.info(f"Raw model output: {repr(decoded)}") 

            
        results = [r.strip().strip(".,;:") for r in decoded.strip().split(",") if r.strip()]
        if not results:
            raise RuntimeError(f"Model returned empty output for {report_id}")

        description_short = results[0] 
        description_synonyms = ",".join(results)  

        logging.info(f"Finished voice_text for {report_id} with result: {results}")

        # Update DB with transcription result
        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            report.description_short = description_short
            report.description_synonyms = description_synonyms
            db.commit()
        finally:
            db.close()
    
    except Exception as e:
        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            report.status = "Failed during voice_text"
            db.commit()
        finally:
            db.close()
        logging.error(f"Voice text failed for {report_id}: {e}")
        raise

    finally:
        try:
            del generation
        except NameError:
            pass
        try:
            del inputs
        except NameError:
            pass
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()