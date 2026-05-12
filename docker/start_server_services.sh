cd /workspace

# Ensure upload directory exists
mkdir -p /data/uploads

# Start Redis
redis-server --daemonize yes

# Start PostgreSQL
service postgresql start

# Start Celery worker in background (solo pool for GPU/CUDA compatibility)
celery -A celery_app worker --loglevel=info --pool=solo \
    --queues=analysis.ingest,analysis.voice_text,analysis.segment,analysis.report &

# Start Flower UI on port 5555 in background
celery -A celery_app flower --port=5555 &

# Start FastAPI
fastapi dev api-server.py --host 0.0.0.0
