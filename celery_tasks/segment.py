import gc
import logging
from db.models import ReportedFrame
import numpy as np
import cv2
import os
import torch
import uuid
import subprocess

from sam3.model_builder import build_sam3_video_predictor
from celery_app import celery_app
from helper import crypto
from db import get_sync_db, Report
from moge.model.v2 import MoGeModel
from geoalchemy2.shape import to_shape


from pyproj import Geod

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

    db = get_sync_db()
    try:
        report = db.get(Report, report_id)
        encrypted_session_key = report.encrypted_session_key
        video_path = report.video_path
        description_short = report.description_short
        description_synonyms = report.description_synonyms
        location_upload = report.location_upload
        orientation_device = report.orientation_device
    finally:
        db.close()

    aes_key = crypto.decrypt_session_key(encrypted_session_key)
    with open(video_path, "rb") as f:
        encrypted_blob = bytearray(f.read())

    plaintext = bytearray(crypto.decrypt_video(aes_key, encrypted_blob))

    mem_path = f"/dev/shm/{uuid.uuid4().hex}.mp4"
    mem_path_rotated = f"/dev/shm/{uuid.uuid4().hex}.mp4"

    try:
        with open(mem_path, "wb") as f:
            f.write(plaintext)

        crypto.shred(plaintext)
        plaintext = None

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mem_path,
                "-metadata:s:v:0", "rotate=0",
                "-c:v", "libx264",
                "-preset", "fast",
                mem_path_rotated
            ],
            check=True,
            capture_output=True
        )
        os.unlink(mem_path)          

        video_predictor = build_sam3_video_predictor()

        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=mem_path_rotated,
            )
        )
        session_id = response["session_id"]

        sam3_total_frames = response.get("num_frames")

        cap = cv2.VideoCapture(mem_path_rotated)
        cv2_total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if cv2_total_frames == 0 or sam3_total_frames == 0:
            raise RuntimeError(f"Could not read video or video has no frames: {mem_path_rotated}")

        total_frames = sam3_total_frames if sam3_total_frames else cv2_total_frames 

        key_frames = [0, total_frames // 4, total_frames // 2, 3 * total_frames // 4, total_frames - 1]

        logging.info(f"Description synonyms: {description_synonyms}")
        raw_terms = [t.strip() for t in description_synonyms.split(",")] if description_synonyms else [description_short]
        
        search_terms = [f"a {t}" if not t.startswith("a ") else t for t in raw_terms]

        any_objects_detected = False
        matched_term = None
        for term in search_terms:
            logging.info(f"Trying search term '{term}' for {report_id}")
            term_detected = False
            for kf in key_frames:
                response = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=kf,
                        text=term,
                    )
                )
                n_objects = len(response["outputs"]["out_obj_ids"])
                logging.info(f"  Term '{term}' frame {kf}: detected {n_objects} objects")
                if n_objects > 0:
                    term_detected = True
                    any_objects_detected = True

            if term_detected:
                matched_term = term
                logging.info(f"Objects found with term '{term}', stopping search")
                break

        if not any_objects_detected:
            logging.warning(f"No objects detected for any search term {search_terms} in {report_id}")

        frames_output = {}
        if any_objects_detected:
            logging.info(f"Propagating masks through video using term '{matched_term}'...")
            for frame_result in video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                frame_idx = frame_result["frame_index"]
                frames_output[frame_idx] = frame_result["outputs"]
            logging.info(f"Processed {len(frames_output)} frames")

        def overlay_masks_with_box(frame, masks, alpha=0.2):
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            vis = frame.copy()

            if len(masks.shape) == 2:
                masks = masks[None, :, :]

            for i, mask in enumerate(masks):
                color = colors[i % len(colors)]
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

                mask_uint8 = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    label = f"Obj {i+1}"
                    cv2.putText(vis, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return vis

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
                if probs is not None and hasattr(probs, 'cpu'):
                    probs = probs.cpu().numpy()

                vis_frame = overlay_masks_with_box(frame, masks)
                mask_area = int((masks > 0).sum())

                if mask_area > 0:
                    sam3_confidence = float(probs.max()) if probs is not None and len(probs) > 0 else 0.0
                    
                    flat_mask = np.any(masks > 0, axis=0) if masks.ndim == 3 else (masks > 0)
                    
                    top_frames.append((mask_area, sam3_confidence, vis_frame.copy(), frame_idx, flat_mask))
                    top_frames.sort(key=lambda x: x[0], reverse=True)
                    top_frames = top_frames[:3]

            frame_idx += 1

        if not top_frames:
            logging.warning(f"No detections found, saving fallback frames for {report_id}")
            for fidx in [0, total_frames // 2, total_frames - 1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if ret:
                    top_frames.append((0, 0.0, frame.copy(), fidx, None))
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_model_path = "metricAnything/student_pointmap.pt"
        logging.info(f"Loading MoGe model from local directory: {local_model_path}")
        model = MoGeModel.from_pretrained(local_model_path).to(device)
        model.eval()

        video_dir = os.path.dirname(video_path)
        safe_desc = "".join(c if c.isalnum() or c in "_-" else "_" for c in (matched_term or description_short))

        frame_distances = {}

        for rank, (mask_area, sam3_confidence, annotated, fidx, flat_mask) in enumerate(top_frames, 1):
            path = os.path.join(video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}.jpg")
            cv2.imwrite(path, annotated)

            logging.info(f"Top {rank}: frame {fidx} (mask_area: {mask_area}px) -> {path}")

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            input_tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)

            with torch.no_grad():
                output = model.infer(input_tensor)

            depth = output["depth"].cpu().numpy()

            median_dist = None
            min_dist = None
            location_segmented = None

            if flat_mask is not None:
                object_depths = depth[flat_mask]
                if len(object_depths) > 0:
                    median_dist = float(np.median(object_depths))
                    min_dist = float(np.min(object_depths))
                    logging.info(f"Target object in frame {fidx} is located at median distance {median_dist:.2f}m, minimum distance {min_dist:.2f}m")
                else:
                    logging.info(f"Target object in frame {fidx} failed to yield valid depth data")


            # calculate location_segmented for each frame with detections based on upload location, orientation and distance median from reported
            if flat_mask is not None:
                location_point = to_shape(report.location_upload)

                geod = Geod(ellps="WGS84")
                print(f"location_upload: {location_upload}, orientation_device: {orientation_device}, median_dist: {median_dist}")
                lon, lat, _ = geod.fwd(lons=location_point.x, lats=location_point.y, az=orientation_device, dist=median_dist)
                location_segmented = f"POINT({lon} {lat})"
                print(f"location_segmented: {location_segmented}")




            frame_distances[fidx] = {"median": median_dist, "min": min_dist, "location_segmented": location_segmented}

            depth_min, depth_max = 0, 50
            depth_clip = np.clip(depth, depth_min, depth_max)
            depth_norm = (depth_clip - depth_min) / (depth_max - depth_min)
            depth_colormap = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

            depth_path = os.path.join(video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}_depth.png")
            cv2.imwrite(depth_path, depth_colormap)
            logging.info(f"Depth map saved -> {depth_path}")




        db = get_sync_db()
        try:
            report = db.get(Report, report_id)

            for rank, (mask_area, sam3_confidence, annotated, fidx, flat_mask) in enumerate(top_frames, 1):
                height, width = annotated.shape[:2]
                total_pixels = height * width
                mask_coverage = float(mask_area / total_pixels) if total_pixels > 0 else 0.0

                dist_data = frame_distances.get(fidx, {"median": None, "min": None, "location_segmented": None})

                new_frame = ReportedFrame(
                    report_id=report_id,
                    image_path=os.path.join(video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}.jpg"),
                    confidence=sam3_confidence,
                    mask_coverage=mask_coverage,
                    distance_median_from_reported_location=dist_data["median"],
                    distance_min_from_reported_location=dist_data["min"],
                    location_segmented=dist_data["location_segmented"]
                )
                report.frames.append(new_frame)

            report.status = "segmented" if any_objects_detected  else "segmented_no_detection"
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

        logging.info(f"Done! Output dir: {video_dir}")

    except Exception as e:
        db = get_sync_db()
        try:
            report = db.get(Report, report_id)
            report.status = "Failed during segmentation"
            db.commit()
        finally:
            db.close()
        logging.error(f"Segmentation failed for {report_id}: {e}")
        raise

    finally:
        if 'video_predictor' in locals() and 'session_id' in locals():
            try:
                video_predictor.handle_request(dict(type="close_session", session_id=session_id))
            except Exception:
                pass
        
        if 'video_predictor' in locals():
            try:
                del video_predictor
            except Exception:
                pass

        if 'cap' in locals() and cap.isOpened():
            cap.release()

        try:
            crypto.shred(aes_key)
        except NameError:
            pass
        try:
            crypto.shred(encrypted_blob)
        except NameError:
            pass

        if plaintext is not None:
            try:
                crypto.shred(plaintext)
            except NameError:
                pass

        if os.path.exists(mem_path):
            os.unlink(mem_path)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    logging.info(f"Finished segment for {report_id}")