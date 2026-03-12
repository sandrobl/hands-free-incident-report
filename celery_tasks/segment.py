import gc
import logging
from db.models import ReportedFrame
import numpy as np
import cv2
import os
import torch
import uuid, shutil



from sam3.model_builder import build_sam3_video_predictor
from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report


@celery_app.task(
    name="app.tasks.segment.process_upload",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    queue="analysis.segment",
    track_started=True,
)
def process_upload(self, report_id: str, batch_id: str):
    logging.info(f"Segment Processing {report_id} in batch {batch_id}")

    # 1. Fetch encrypted data from DB
    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        encrypted_session_key = report.encrypted_session_key
        video_path = report.video_path
        description_full = report.description_full
        description_short = report.description_short
    finally:
        db.close()

    # 2. Decrypt
    aes_key = crypto.decrypt_session_key(encrypted_session_key)
    encrypted_blob = open(video_path, "rb").read()
    plaintext = crypto.decrypt_video(aes_key, encrypted_blob)

    mem_path = f"/dev/shm/{uuid.uuid4().hex}.mp4"

    try:
        with open(mem_path, "wb") as f:
            f.write(plaintext)

        video_predictor = build_sam3_video_predictor()
        response = video_predictor.handle_request(
            request=dict(type="start_session", resource_path=mem_path)
        )

        # 3. TODO: run SAM3 segmentation, save results to DB
        video_predictor = build_sam3_video_predictor()

        # Start session
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=mem_path,
            )
        )
        session_id = response["session_id"]

        # Get frame count
        cap = cv2.VideoCapture(mem_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Add prompts on key frames (start, 25%, 50%, 75%, end)
        key_frames = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]
        print(f"Adding prompts on key frames: {key_frames}")

        # Track if any objects detected
        any_objects_detected = False
        for kf in key_frames:
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=kf,
                    text=description_short,
                )
            )
            n_objects = len(response["outputs"]["out_obj_ids"])
            print(f"  Frame {kf}: detected {n_objects} objects")
            if n_objects > 0:
                any_objects_detected = True

        # Only propagate if something was detected
        frames_output = {}
        if any_objects_detected:
            print("Propagating masks through video...")
            for frame_result in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = frame_result["frame_index"]
                frames_output[frame_idx] = frame_result["outputs"]
            print(f"Processed {len(frames_output)} frames")
        else:
            print("No objects detected in any key frame. Skipping propagation.")

        def overlay_masks_with_box(frame, masks, alpha=0.4):
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            vis = frame.copy()
            
            if len(masks.shape) == 2:
                masks = masks[None, :, :] # Force 3D dim if single mask
                
            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
                
                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Get the largest contour (main object part)
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"Obj {i+1}"
                    cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return vis

        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Track top 3 frames by confidence
        top_frames = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frames_output:
                output = frames_output[frame_idx]
                masks = output["out_binary_masks"]
                probs = output["out_probs"]
                
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
                if hasattr(probs, 'cpu'):
                    probs = probs.cpu().numpy()
                
                vis_frame = overlay_masks_with_box(frame, masks)        

                # Track top 3
                if len(probs):
                    max_conf = probs.max()
                    top_frames.append((max_conf, vis_frame.copy(), frame_idx))
                    top_frames.sort(key=lambda x: x[0], reverse=True)
                    top_frames = top_frames[:3]
            else:
                vis_frame = frame
            
            out_video.write(vis_frame)
            frame_idx += 1

        cap.release()
        out_video.release()

        # Save top 3 frames
        video_dir = os.path.dirname(video_path)

        for rank, (conf, annotated, fidx) in enumerate(top_frames, 1):
            path = os.path.join(video_dir, f"_top{rank}_{description_short}_frame_{fidx:04d}.jpg")
            cv2.imwrite(path, annotated)
            print(f"Top {rank}: frame {fidx} (confidence: {conf:.3f}) -> {path}")

        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            
            for rank, (conf, annotated, fidx) in enumerate(top_frames, 1):
                new_frame = ReportedFrame(
                    report_id=report_id,
                    image_path=os.path.join(video_dir, f"_top{rank}_{description_short}_frame_{fidx:04d}.jpg"),
                    confidence=float(conf)
                )
                report.frames.append(new_frame)
                report.status = "segmented"
                db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
        print(f"\nDone! Output video: {video_dir}")

   
    finally:
        crypto.shred(plaintext)
        if os.path.exists(mem_path):
            os.unlink(mem_path)
    
     # 4. Shred sensitive material
    crypto.shred(aes_key)
    crypto.shred(plaintext)
    
    torch.cuda.empty_cache()
    gc.collect()

    logging.info(f"Finished segment for {report_id}")