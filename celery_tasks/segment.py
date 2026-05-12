import gc
import logging
from db.models import ReportedFrame
import numpy as np
import cv2
import os
import torch
import uuid
import subprocess
import math
import time

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
    start_time = time.time()

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

    # mem_path = f"/tmp/{uuid.uuid4().hex}.mp4"
    # mem_path_rotated = f"/tmp/{uuid.uuid4().hex}.mp4"
    # mem_path_sam3 = f"/tmp/{uuid.uuid4().hex}_sam3.mp4"

    mem_path         = f"/dev/shm/{uuid.uuid4().hex}.mp4"
    mem_path_rotated = f"/dev/shm/{uuid.uuid4().hex}.mp4"
    mem_path_sam3    = f"/dev/shm/{uuid.uuid4().hex}_sam3.mp4"

    keep_every = 1
    session_id = None
    video_predictor = None
    actual_proxy_frames = 0
    cap = None

    segmentation_word = None

    SAM3_MAX_FRAMES = 400
    SAM3_MAX_WIDTH  = 640

    try:
        with open(mem_path, "wb") as f:
            f.write(plaintext)

        crypto.shred(plaintext)
        plaintext = None

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mem_path,
                "-metadata:s:v:0", "rotate=0",
                "-c:v", "libx264",
                "-preset", "fast",
                mem_path_rotated,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg rotation failed:\n{result.stderr.decode('utf-8', errors='replace')}"
            )
        os.unlink(mem_path)

        cap_probe = cv2.VideoCapture(mem_path_rotated)
        src_fps    = cap_probe.get(cv2.CAP_PROP_FPS) or 30
        src_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_probe.release()

        keep_every = max(1, math.ceil(src_frames / SAM3_MAX_FRAMES))
        proxy_fps  = max(1.0, src_fps / keep_every)

        vf = (
            f"scale={SAM3_MAX_WIDTH}:-2"
            if keep_every == 1
            else f"fps={proxy_fps},scale={SAM3_MAX_WIDTH}:-2"
        )

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mem_path_rotated,
                "-vf", vf,
                "-an",
                "-c:v", "libx264", "-preset", "fast",
                mem_path_sam3,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg proxy failed:\n{result.stderr.decode('utf-8', errors='replace')}"
            )

        cap_verify = cv2.VideoCapture(mem_path_sam3)
        actual_proxy_frames = int(cap_verify.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_verify.release()
        logging.info(
            f"SAM3 proxy: 1 in every {keep_every} frames \n "
            f"{actual_proxy_frames} frames @ {proxy_fps:.1f} fps, "
            f"max {SAM3_MAX_WIDTH}px wide"
        )

        video_predictor = build_sam3_video_predictor()

        model_class = video_predictor.model.__class__
        if not hasattr(model_class, "_is_patched"):
            _orig = model_class.init_state

            def patched_init_state(self, *args, **kwargs):
                kwargs.pop("offload_state_to_cpu", None)
                kwargs.pop("offload_video_to_cpu", None)
                kwargs.pop("async_loading_frames", None)
                return _orig(self, *args, **kwargs)

            model_class.init_state = patched_init_state
            model_class._is_patched = True

        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=mem_path_sam3,
            )
        )
        session_id = response["session_id"]
        sam3_total_frames = response.get("num_frames") or actual_proxy_frames

        cap = cv2.VideoCapture(mem_path_rotated)
        cv2_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cv2_total_frames == 0:
            logging.warning(
                "OpenCV could not read frame count from rotated video, "
                "falling back to proxy frame count * keep_every"
            )
            cv2_total_frames = actual_proxy_frames * keep_every

        total_frames = sam3_total_frames if sam3_total_frames else actual_proxy_frames
        key_frames = [
            0,
            total_frames // 4,
            total_frames // 2,
            3 * total_frames // 4,
            total_frames - 1,
        ]

        logging.info(f"Description synonyms: {description_synonyms}")
        raw_terms = (
            [t.strip() for t in description_synonyms.split(",")]
            if description_synonyms
            else [description_short]
        )
        search_terms = [f"a {t}" if not t.startswith("a ") else t for t in raw_terms]

        any_objects_detected = False
        matched_term = None
        total_search_detections = 0
        stored_search_frames = []
        skip_propagation = False

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
                outputs = response.get("outputs", {})
                obj_ids = outputs.get("out_obj_ids", [])
                n_objects = len(obj_ids)
                
                logging.info(f"  Term '{term}' frame {kf}: detected {n_objects} objects")
                if n_objects > 0:
                    term_detected = True
                    any_objects_detected = True
                    total_search_detections += n_objects
                    
                    masks = outputs.get("out_binary_masks")
                    probs = outputs.get("out_probs")
                    if masks is not None:
                        stored_search_frames.append((kf, masks, probs))

            if term_detected:
                matched_term = term
                segmentation_word = term
                logging.info(f"Objects found with term '{term}', stopping search")
                if total_search_detections > 10:
                    skip_propagation = True
                    logging.info("Threshold exceeded in search phase. Skipping propagation.")
                break

        if not any_objects_detected:
            logging.warning(
                f"No objects detected for any search term {search_terms} in {report_id}"
            )

        def overlay_masks_with_box(frame, masks, alpha=0.2):
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
            vis = frame.copy()
            h_dst, w_dst = frame.shape[:2]

            if len(masks.shape) == 2:
                masks = masks[None, :, :]

            for i, mask in enumerate(masks):
                mask_resized = cv2.resize(
                    (mask > 0).astype(np.uint8),
                    (w_dst, h_dst),
                    interpolation=cv2.INTER_NEAREST,
                )

                color = colors[i % len(colors)]
                overlay = vis.copy()
                overlay[mask_resized > 0] = color
                vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

                mask_uint8 = mask_resized * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        vis, f"Obj {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                    )

            return vis

        top_frames = []

        if any_objects_detected:
            if skip_propagation:
                logging.info("Bypassing propagation and using search phase frames directly.")
                for proxy_idx, masks, probs in stored_search_frames:
                    if hasattr(masks, "cpu"):
                        masks = masks.cpu().numpy()
                    if probs is not None and hasattr(probs, "cpu"):
                        probs = probs.cpu().numpy()

                    src_idx = min(proxy_idx * keep_every, cv2_total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    h_src, w_src = frame.shape[:2]
                    if masks.ndim == 3:
                        resized = np.stack([
                            cv2.resize(
                                (m > 0).astype(np.uint8),
                                (w_src, h_src),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            for m in masks
                        ])
                    else:
                        resized = cv2.resize(
                            (masks > 0).astype(np.uint8),
                            (w_src, h_src),
                            interpolation=cv2.INTER_NEAREST,
                        )[None]

                    mask_area = int((resized > 0).sum())
                    if mask_area == 0:
                        continue

                    flat_mask = np.any(resized > 0, axis=0)
                    vis_frame = overlay_masks_with_box(frame, masks)
                    sam3_confidence = (
                        float(probs.max()) if probs is not None and len(probs) > 0 else 0.0
                    )

                    top_frames.append(
                        (mask_area, sam3_confidence, vis_frame, src_idx, flat_mask)
                    )
                
                top_frames.sort(key=lambda x: x[0], reverse=True)
                top_frames = top_frames[:3]

            else:
                logging.info(f"Propagating masks through video using term '{matched_term}'")
                frame_count = 0
                for frame_result in video_predictor.handle_stream_request(
                    request=dict(type="propagate_in_video", session_id=session_id)
                ):
                    proxy_idx = frame_result["frame_index"]
                    output = frame_result["outputs"]
                    frame_count += 1

                    masks = output["out_binary_masks"]
                    probs = output["out_probs"]

                    if hasattr(masks, "cpu"):
                        masks = masks.cpu().numpy()
                    if probs is not None and hasattr(probs, "cpu"):
                        probs = probs.cpu().numpy()

                    if masks.ndim == 3 and masks.shape[0] == 0:
                        continue

                    src_idx = min(proxy_idx * keep_every, cv2_total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    h_src, w_src = frame.shape[:2]
                    if masks.ndim == 3:
                        resized = np.stack([
                            cv2.resize(
                                (m > 0).astype(np.uint8),
                                (w_src, h_src),
                                interpolation=cv2.INTER_NEAREST,
                            )
                            for m in masks
                        ])
                    else:
                        resized = cv2.resize(
                            (masks > 0).astype(np.uint8),
                            (w_src, h_src),
                            interpolation=cv2.INTER_NEAREST,
                        )[None]

                    mask_area = int((resized > 0).sum())
                    if mask_area == 0:
                        continue

                    flat_mask = np.any(resized > 0, axis=0)
                    vis_frame = overlay_masks_with_box(frame, masks)
                    sam3_confidence = (
                        float(probs.max()) if probs is not None and len(probs) > 0 else 0.0
                    )

                    top_frames.append(
                        (mask_area, sam3_confidence, vis_frame, src_idx, flat_mask)
                    )
                    top_frames.sort(key=lambda x: x[0], reverse=True)
                    top_frames = top_frames[:3]

                logging.info(f"Processed {frame_count} frames")

        if video_predictor is not None and session_id is not None:
            try:
                video_predictor.handle_request(
                    dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass
            session_id = None

        if video_predictor is not None:
            del video_predictor
            video_predictor = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        if os.path.exists(mem_path_sam3):
            os.unlink(mem_path_sam3)

        if not top_frames:
            logging.warning(
                f"No detections found, saving fallback frames for {report_id}"
            )
            for fidx in [0, cv2_total_frames // 2, cv2_total_frames - 1]:
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
        safe_desc = "".join(
            c if c.isalnum() or c in "_-" else "_"
            for c in (matched_term or description_short)
        )

        frame_distances = {}

        for rank, (mask_area, sam3_confidence, annotated, fidx, flat_mask) in enumerate(
            top_frames, 1
        ):
            path = os.path.join(
                video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}.jpg"
            )
            cv2.imwrite(path, annotated)
            logging.info(f"Top {rank}: frame {fidx} (mask_area: {mask_area}px) -> {path}")

            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            input_tensor = torch.tensor(
                frame_rgb / 255.0, dtype=torch.float32, device=device
            ).permute(2, 0, 1)

            with torch.no_grad():
                output = model.infer(input_tensor)

            depth = output["depth"].cpu().numpy()

            del output
            del input_tensor
            torch.cuda.empty_cache()

            median_dist = None
            min_dist = None
            location_segmented = None

            if flat_mask is not None:
                object_depths = depth[flat_mask]
                if len(object_depths) > 0:
                    median_dist = float(np.median(object_depths))
                    min_dist = float(np.min(object_depths))
                    logging.info(
                        f"Target object in frame {fidx} is at median distance "
                        f"{median_dist:.2f}m, minimum distance {min_dist:.2f}m"
                    )
                else:
                    logging.info(
                        f"Target object in frame {fidx} failed to yield valid depth data"
                    )

            if flat_mask is not None and median_dist is not None and orientation_device is not None:
                location_point = to_shape(location_upload)
                geod = Geod(ellps="WGS84")
                logging.info(
                    f"location_upload: {location_upload}, "
                    f"orientation_device: {orientation_device}, "
                    f"median_dist: {median_dist}"
                )
                lon, lat, _ = geod.fwd(
                    lons=location_point.x,
                    lats=location_point.y,
                    az=orientation_device,
                    dist=median_dist,
                )
                location_segmented = f"POINT({lon} {lat})"
                logging.info(f"location_segmented: {location_segmented}")
            elif flat_mask is not None and median_dist is not None:
                logging.warning(
                    f"Skipping location_segmented for frame {fidx}: orientation_device is None"
                )

            frame_distances[fidx] = {
                "median": median_dist,
                "min": min_dist,
                "location_segmented": location_segmented,
            }

            depth_clip = np.clip(depth, 0, 50)
            depth_norm = depth_clip / 50.0
            depth_colormap = cv2.applyColorMap(
                (depth_norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA
            )
            depth_path = os.path.join(
                video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}_depth.png"
            )
            cv2.imwrite(depth_path, depth_colormap)
            logging.info(f"Depth map saved -> {depth_path}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        db = get_sync_db()
        try:
            report = db.get(Report, report_id)

            for rank, (mask_area, sam3_confidence, annotated, fidx, flat_mask) in enumerate(
                top_frames, 1
            ):
                height, width = annotated.shape[:2]
                total_pixels = height * width
                mask_coverage = (
                    float(mask_area / total_pixels) if total_pixels > 0 else 0.0
                )

                dist_data = frame_distances.get(
                    fidx, {"median": None, "min": None, "location_segmented": None}
                )

                new_frame = ReportedFrame(
                    report_id=report_id,
                    image_path=os.path.join(
                        video_dir, f"_top{rank}_{safe_desc}_frame_{fidx:04d}.jpg"
                    ),
                    confidence=sam3_confidence,
                    mask_coverage=mask_coverage,
                    distance_median_from_reported_location=dist_data["median"],
                    distance_min_from_reported_location=dist_data["min"],
                    location_segmented=dist_data["location_segmented"],
                )
                report.frames.append(new_frame)

            report.status = (
                "segmented" if any_objects_detected else "segmented_no_detection"
            )

            report.segmented_word = segmentation_word

            report.segment_duration = time.time() - start_time

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
        if video_predictor is not None and session_id is not None:
            try:
                video_predictor.handle_request(
                    dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass

        if video_predictor is not None:
            try:
                del video_predictor
            except Exception:
                pass

        if cap is not None and cap.isOpened():
            cap.release()

        for tmp_path in [mem_path_sam3, mem_path_rotated]:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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