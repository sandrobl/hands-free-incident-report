import json
from typing import cast
import uuid
import os
import pathlib
import shutil
import requests


from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = pathlib.Path("/data/uploads")

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from cryptography.hazmat.primitives import serialization
from contextlib import asynccontextmanager

from celery import chain
from celery_app import celery_app

from db import Base, engine, get_db, Report
from fastapi import Depends

from fastapi_plugin.fast_api_client import Auth0FastAPI
from fastapi.security import HTTPBearer

from sqlalchemy.ext.asyncio import AsyncSession
from geoalchemy2 import Geometry
from sqlalchemy.orm import selectinload
from sqlalchemy import select, func


@asynccontextmanager
async def lifespan(app: FastAPI):
    private_key_pem = os.environ["RSA_PRIVATE_KEY"].encode("utf-8")

    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
    )

    public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("utf-8")

    app.state.private_key = private_key
    app.state.public_key = public_key_pem

    app.state.auth0_domain = os.environ["AUTH0_DOMAIN"]
    app.state.auth0_audience = os.environ["AUTH0_API_AUDIENCE"]
    app.state.auth0_location = os.environ["AUTH0_API_LOCATION"]

    # Initialize Auth0FastAPI and attach to app.state
    app.state.auth0 = Auth0FastAPI(
        domain=app.state.auth0_domain,
        audience=app.state.auth0_audience,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    del app.state.private_key
    del app.state.public_key
    del app.state.auth0_domain
    del app.state.auth0_audience
    del app.state.auth0



security = HTTPBearer()

app = FastAPI(lifespan=lifespan)

class TokenRequest(BaseModel):
    client_id: str
    client_secret: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/new_report/{user_id}")
async def new_report(user_id: str, db: AsyncSession = Depends(get_db)):
    report = Report(user_id=user_id)
    db.add(report)
    await db.flush()
    await db.refresh(report)
    await db.commit()

    return {
        "report_id": report.report_id,
        "public_key": app.state.public_key,
        "expires_at": "2026-12-12-12:00:00Z"
    }

@app.post("/upload_video/")
async def receive_encrypted_video(
    report_id: str,
    encrypted_session_key: str = Form(),
    encrypted_video: UploadFile = File(),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    print(f"Received upload for report_id: {report_id}, filename: {encrypted_video.filename}, content_type: {encrypted_video.content_type}")
    # Validate report_id is a UUID to prevent path traversal
    try:
        uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report_id")

    report = await db.get(Report, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    # Save encrypted video to disk
    upload_path = UPLOAD_DIR / report_id
    upload_path.mkdir(parents=True, exist_ok=True)
    video_path = upload_path / (report_id + "-video.enc")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(encrypted_video.file, buffer)

    # Store session key and video path in DB
    report.encrypted_session_key = encrypted_session_key
    report.video_path = str(video_path)

    if latitude is not None and longitude is not None:
        report.location = f"SRID=4326;POINT({longitude} {latitude})"
    await db.commit()

    batch_id = str(uuid.uuid4())
    result = chain(
        celery_app.signature("app.tasks.ingest.process_upload",     args=[report_id, batch_id], immutable=True),
        celery_app.signature("app.tasks.voice_text.process_upload", args=[report_id, batch_id], immutable=True),
        celery_app.signature("app.tasks.segment.process_upload",    args=[report_id, batch_id], immutable=True),
        celery_app.signature("app.tasks.report.process_upload",     args=[report_id, batch_id], immutable=True),
    ).apply_async()

    return {
        "message": "Video uploaded and pipeline started",
        "report_id": report_id,
        "batch_id": batch_id,
        "task_id": result.id,
    }


@app.get("/report/{report_id}")
async def get_report(report_id: str, db: AsyncSession = Depends(get_db)):
    stmt = (
            select(
                Report,
                func.ST_AsGeoJSON(Report.location).label("location_json")
            )
            .options(selectinload(Report.frames))
            .where(Report.report_id == report_id)
        )
    
    result = await db.execute(stmt)
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report = await db.get(Report, report_id, options=[selectinload(Report.frames)])
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    location_data = None
    if row.location_json:
        location_data = json.loads(row.location_json)

    return {
        "report_id": report.report_id,
        "description_full": report.description_full,
        "description_short": report.description_short,
        "location": location_data,
        "created_at": report.created_at.isoformat(),
        "reported_frames": [
            {
                "reported_frame_id": frame.reported_frame_id,
                "image_path": frame.image_path,
                "confidence": frame.confidence
            } for frame in report.frames
        ]
    }

@app.get("/api/public")
def public():
   # No access token required to access this route

   result = {
       "status": "success",
       "msg": ("Hello from a public endpoint! You don't need to be "
               "authenticated to see this.")
   }
   return result


@app.get("/api/private")
def private(
    claims: dict = Depends(lambda: app.state.auth0.require_auth()()),
    token: str = Depends(security)
):
    # A valid access token is required to access this route
    return claims

@app.post("/api/token")
def get_token(body: TokenRequest):
    response = requests.post(
        f"{app.state.auth0_location}/oauth/token",
        headers={"content-type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": body.client_id,
            "client_secret": body.client_secret,
            "audience": app.state.auth0_audience,
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())

    return response.json()