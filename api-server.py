import json
from typing import cast
import uuid
import os
import pathlib
import shutil
import requests
from dateutil import parser
from datetime import timezone


from dotenv import load_dotenv
load_dotenv()

UPLOAD_DIR = pathlib.Path("/data/uploads")

from fastapi import Depends
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi_plugin.fast_api_client import Auth0FastAPI
from fastapi.security import HTTPBearer
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from cryptography.hazmat.primitives import serialization
from contextlib import asynccontextmanager

from celery import chain
from celery_app import celery_app

from db import Base, engine, get_db, Report, ReportedFrame


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "https://admin.hands-free-incident-report.ch", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/public_key")
async def get_public_key():
    return PlainTextResponse(app.state.public_key)


class TokenRequest(BaseModel):
    client_id: str
    client_secret: str


@app.post("/new_report/")
async def new_report(request: Request, db: AsyncSession = Depends(get_db)):
    claims = await app.state.auth0.require_auth()(request)

    report = Report(user_id=claims["sub"])
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
    request: Request,
    report_id: str,
    encrypted_session_key: str = Form(),
    encrypted_video: UploadFile = File(),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    orientation: str | None = Form(None),
    created_at: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    claims = await app.state.auth0.require_auth()(request)

    print(f"Received upload for report_id: {report_id}, filename: {encrypted_video.filename}, content_type: {encrypted_video.content_type}")
    print(f"Form data - encrypted_session_key: {encrypted_session_key}, latitude: {latitude}, longitude: {longitude}, orientation: {orientation}, created_at: {created_at}")
    # Validate report_id is a UUID to prevent path traversal
    try:
        uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report_id")

    report = await db.get(Report, report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != claims["sub"]:
        raise HTTPException(status_code=403, detail="Not your report")

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
        report.location_upload = f"SRID=4326;POINT({longitude} {latitude})"
    if orientation is not None:
        report.orientation_device = orientation
    if created_at is not None:
        report.created_at = parser.parse(created_at).replace(tzinfo=timezone.utc)
                
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
async def get_report(request: Request, report_id: str, db: AsyncSession = Depends(get_db)):
    claims = await app.state.auth0.require_auth()(request)

    stmt = (
        select(
            Report,
            func.ST_AsGeoJSON(Report.location_upload).label("location_json"),
            ReportedFrame,
            func.ST_AsGeoJSON(ReportedFrame.location_segmented).label("frame_location_json")
        )
        .outerjoin(Report.frames)
        .where(Report.report_id == report_id)
    )

    result = await db.execute(stmt)
    rows = result.all()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Report not found")

    report, location_json = rows[0].Report, rows[0].location_json  # unpack directly
    location_upload_data = json.loads(location_json) if location_json else None

    duplicates_result = await db.execute(
        select(Report.report_id).where(Report.duplicate_of == report_id)
    )
    
    duplicates = duplicates_result.scalars().all()

    return {
        "report_id": report.report_id,
        "description_full": report.description_full,
        "description_short": report.description_short,
        "description_synonyms": report.description_synonyms,
        "segmented_word": report.segmented_word,
        "orientation": report.orientation_device,
        "duplicate_of": report.duplicate_of,
        "duplicate_confidence": report.duplicate_confidence,
        "duplicates": duplicates,
        "status": report.status,
        "location_upload": location_upload_data,
        "created_at": report.created_at.isoformat(),
        "reported_frames": [
            {
                "reported_frame_id": row.ReportedFrame.reported_frame_id,
                "image_path": row.ReportedFrame.image_path,
                "frame_url": f"/report/{report_id}/frames/{os.path.basename(row.ReportedFrame.image_path)}",
                "confidence": row.ReportedFrame.confidence,
                "mask_coverage": row.ReportedFrame.mask_coverage,
                "location_segmented": json.loads(row.frame_location_json) if row.frame_location_json else None,
                "distance_median_from_reported_location": row.ReportedFrame.distance_median_from_reported_location,
            } for row in rows if row.ReportedFrame is not None
        ]
    }

@app.get("/report/{report_id}/frames/{filename}")
async def get_frame(
    request: Request,
    report_id: str,
    filename: str,
    db: AsyncSession = Depends(get_db)
):
    claims = await app.state.auth0.require_auth()(request)

    # Prevent path traversal
    try:
        uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid report_id")
    
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = UPLOAD_DIR / report_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(file_path, media_type="image/jpeg")


@app.get("/reports")
async def get_reports(
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    claims = await app.state.auth0.require_auth()(request)
    result = await db.execute(select(Report))
    reports = result.scalars().all()
    return [{"report_id": r.report_id, "status": r.status, "created_at": r.created_at.isoformat()} for r in reports]

@app.get("/api/private")
async def private(request: Request, token: str = Depends(security)):
    claims = await app.state.auth0.require_auth()(request)
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

@app.get("/privacy", response_class=HTMLResponse)
async def get_privacy_policy():
    html_content = """
    <!DOCTYPE html>
    <html lang="de">
    <head>
        <meta charset="UTF-8">
        <title>Datenschutzerklärung - Master Thesis</title>
        <style>
            body { font-family: sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }
            h1 { color: #333; }
            h2 { color: #555; }
        </style>
    </head>
    <body>
        <h1>Datenschutzerklärung</h1>
        <p>Diese App wird im Rahmen einer Masterarbeit an der Universität St. Gallen (HSG) entwickelt.</p>
        
        <h2>1. Verantwortlicher</h2>
        <p>S.Blatter<br>Universität St. Gallen (HSG)<br></p>
        
        <h2>2. Zugriff auf Kamera und Mikrofon</h2>
        <p>Die App nutzt die Kamera und das Mikrofon der verbundenen Smartglasses ausschließlich zur Erstellung von Unfallberichten. Die Aufnahme startet nur nach expliziter Nutzerinteraktion.</p>
        
        <h2>3. Datenspeicherung und Verarbeitungszweck</h2>
        <p>Sämtliche Audio und Videodaten dienen rein wissenschaftlichen Zwecken im Rahmen der Master Thesis. Die Daten werden verschlüsselt an den Server des Projekts übertragen oder lokal gespeichert.</p>
        
        <h2>4. Weitergabe an Dritte</h2>
        <p>Es erfolgt keine Weitergabe von personenbezogenen Daten an Dritte oder kommerzielle Organisationen. Meta (als Hardwarehersteller) erhält durch diese App keinen Zugriff auf die aufgenommenen Rohdaten der Berichte.</p>
        
        <h2>5. Nutzerrechte</h2>
        <p>Nutzer haben das Recht auf Auskunft, Korrektur oder Löschung ihrer während der Testphase erhobenen Daten.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)
