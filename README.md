# handsfree-incident-report

Hands-free incident report pipeline built around FastAPI, Celery, Redis, and PostgreSQL/PostGIS. The server accepts encrypted video uploads, runs a multi-step analysis pipeline, and serves report data and selected frames.

## What This Repository Contains

- FastAPI backend in [api-server.py](api-server.py) with Auth0-protected endpoints and a built-in privacy policy page.
- Celery tasks for ingest, voice-to-text optimization, segmentation, and report correlation in [celery_tasks](celery_tasks).
- PostgreSQL models and DB helpers in [db](db).
- Docker and setup scripts in [docker](docker).
- Dataset analytics and summaries in [dataset](dataset).

## Pipeline Overview (Celery)

The upload endpoint triggers a Celery chain in this order:

1. **Ingest**: Decrypts the video, extracts audio via `ffmpeg`, transcribes with Whisper `large-v3`, and stores `description_full`. See [celery_tasks/ingest.py](celery_tasks/ingest.py).
2. **Voice Text**: Uses a Gemma model to extract key noun phrases (object-focused) and stores `description_short` and `description_synonyms`. See [celery_tasks/voice_text.py](celery_tasks/voice_text.py).
3. **Segment**: Runs SAM3 video segmentation using text prompts, stores top frames, computes depth with MoGe, and optionally derives a segmented location. See [celery_tasks/segment.py](celery_tasks/segment.py).
4. **Report**: Checks for nearby duplicates in the last 7 days using sentence embeddings and updates status/duplicate fields. See [celery_tasks/report.py](celery_tasks/report.py).

Celery is configured in [celery_app.py](celery_app.py) with Redis as broker and backend, and queues:

- `analysis.ingest`
- `analysis.voice_text`
- `analysis.segment`
- `analysis.report`

## API Endpoints

Implemented in [api-server.py](api-server.py):

- `GET /public_key` — returns the RSA public key.
- `POST /upload_video/` — accepts encrypted video upload and metadata, then starts the Celery chain.
- `GET /report/{report_id}` — returns report details and reported frames.
- `GET /report/{report_id}/frames/{filename}` — returns a saved frame image.
- `GET /reports` — lists report IDs, status, and timestamps.
- `GET /api/private` — Auth0-protected endpoint returning token claims.
- `POST /api/token` — fetches Auth0 client-credentials token.
- `GET /privacy` — privacy policy HTML (German) for the master thesis app.

Uploads are stored under `/data/uploads` (see [api-server.py](api-server.py)).

## Database Schema

Defined in [db/models.py](db/models.py):

- `reports` table with fields for encrypted key, paths, text descriptions, location, status, and duration metrics.
- `reported_frame` table with image path, mask coverage, confidence, distance, and segmented location.

The DB helpers and engines are defined in [db/database.py](db/database.py).

## Configuration (Environment Variables)

The server and tasks expect these environment variables (see [api-server.py](api-server.py) and [db/database.py](db/database.py)):

- `RSA_PRIVATE_KEY`
- `AUTH0_DOMAIN`
- `AUTH0_API_AUDIENCE`
- `AUTH0_API_LOCATION`
- `DATABASE_URL`

The Docker setup script also references:

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_DB`
- `HUGGING_FACE_HUB_TOKEN` (optional)

## Scripts

- [docker/setup.sh](docker/setup.sh): installs system dependencies, installs ML/vision stacks, initializes PostgreSQL extensions, and starts services.
- [docker/start_server_services.sh](docker/start_server_services.sh): starts Redis, PostgreSQL, Celery worker, Flower, and FastAPI.
- [helper/recreate_tables.py](helper/recreate_tables.py): drops and recreates DB tables.
- [helper/download_models.py](helper/download_models.py): downloads the Gemma model into `/workspace/gemma-4`.
- [helper/merge-audio.ps1](helper/merge-audio.ps1): combines a visual file with audio using `ffmpeg`.

## Dataset and Analytics

The [dataset](dataset) folder includes:

- [dataset/Results.csv](dataset/Results.csv) and a report generator script [dataset/generate_report.py](dataset/generate_report.py).
- Chart summaries in [dataset/chart_summaries.md](dataset/chart_summaries.md).
- A sample list of descriptions in [dataset/description.md](dataset/description.md).

## Notes on Data Flow

- Encrypted uploads are saved under `/data/uploads/{report_id}`.
- The ingest task uses Whisper `large-v3` with audio extracted via `ffmpeg`.
- Segmentation uses SAM3 and writes annotated frames and depth maps alongside the uploaded video.
- Duplicate detection checks recent reports within 10 meters and compares sentence embeddings.
