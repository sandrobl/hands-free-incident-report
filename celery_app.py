from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

celery_app = Celery(
    "handsfree",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

celery_app.conf.update(
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    task_routes={
        "app.tasks.ingest.*":     {"queue": "analysis.ingest"},
        "app.tasks.voice_text.*": {"queue": "analysis.voice_text"},
        "app.tasks.segment.*":    {"queue": "analysis.segment"},
        "app.tasks.report.*":     {"queue": "analysis.report"},
    },
)

# Must be after celery_app is defined — task files import celery_app
import celery_tasks.ingest       
import celery_tasks.voice_text   
import celery_tasks.segment      
import celery_tasks.report       