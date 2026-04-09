#!/bin/bash
set -e

SETUP_MARKER="/workspace/.sam3_setup_done"

if [ ! -f "$SETUP_MARKER" ]; then

    # ========================================
    # 1. System packages
    # ========================================
    export DEBIAN_FRONTEND=noninteractive
    export TZ=Etc/UTC
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

    # Bootstrap tools, then add PostgreSQL 17 repo
    apt update
    apt install -y gnupg lsb-release curl

    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/trusted.gpg.d/postgresql.gpg
    echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list

    apt update
    apt install -y tzdata python3.12 python3-pip git \
        libgl1 libglib2.0-0 redis-server wget openssl \
        postgresql-17 postgresql-17-postgis-3 postgresql-17-postgis-3-scripts \
        postgresql-server-dev-17 build-essential ffmpeg build-essential cmake \
        ninja-build libprotobuf-dev protobuf-compiler

    ln -sf /usr/bin/python3.12 /usr/bin/python

    # ========================================
    # 2. PGVector (build from source)
    # ========================================
    cd /tmp
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
    cd pgvector && make && make install
    cd / && rm -rf /tmp/pgvector

    # # ========================================
    # # 3. PyTorch + Hugging Face
    # # ========================================
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --break-system-packages
    pip install --upgrade huggingface_hub --break-system-packages

    if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
        python -c "from huggingface_hub import login; login(token='$HUGGING_FACE_HUB_TOKEN')"
    else
        echo 'WARNING: HUGGING_FACE_HUB_TOKEN not set.'
    fi

    python -c 'import torch; print("CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'

    # # ========================================
    # # 4. SAM3
    # # ========================================
    cd /workspace
    if [ ! -d 'sam3' ]; then
        git clone https://github.com/facebookresearch/sam3.git
    fi
    cd sam3
    pip install . '.[notebooks]' '.[train,dev]' --break-system-packages
    python -c "import sam3; print('SAM3', sam3.__version__)"


    # # ========================================
    # # Whisper
    # # ========================================
    pip install -U openai-whisper --break-system-packages
    pip install setuptools-rust --break-system-packages


    # # ========================================
    # # Auth0 FastAPI
    # # ========================================
    pip install auth0-fastapi-api --break-system-packages

    # # ========================================
    # # Gemma
    # # ========================================
    pip install git+https://github.com/huggingface/transformers.git --break-system-packages
    pip install accelerate --break-system-packages

    # # ========================================
    # # Sentence Transformers
    # # ========================================
    pip install sentence-transformers --break-system-packages

    # # =======================================
    # # Metric Anything
    # # =======================================
    pip install git+https://github.com/microsoft/MoGe.git --break-system-packages

    # # =======================================
    # # Pyproj for geospatial calculations
    # # =======================================
    pip install pyproj --break-system-packages

    # ========================================
    # 5. Python app dependencies
    # ========================================
    pip install "fastapi[standard]" celery[redis] flower redis cryptography python-dotenv \
        "psycopg[binary,pool]" pgvector sqlalchemy[asyncio] alembic geoalchemy2 geoalchemy2[shapely] --break-system-packages

    # ========================================
    # 6. Storage + PostgreSQL DB init
    # ========================================
    mkdir -p /data/uploads

    PG_CONF="/etc/postgresql/17/main/postgresql.conf"
    PG_HBA="/etc/postgresql/17/main/pg_hba.conf"
    sed -i "s/^#\?listen_addresses\s*=.*/listen_addresses = '*'/" "$PG_CONF"
    if ! grep -q '^host all all 0.0.0.0/0 scram-sha-256$' "$PG_HBA"; then
        echo 'host all all 0.0.0.0/0 scram-sha-256' >> "$PG_HBA"
    fi

    service postgresql start

    PG_PASSWORD=$POSTGRES_PASSWORD
    PG_USER=$POSTGRES_USER
    PG_DB=$POSTGRES_DB

    su postgres -c "psql -c \"CREATE USER ${PG_USER} WITH PASSWORD '${PG_PASSWORD}';\"" || true
    su postgres -c "psql -c \"CREATE DATABASE ${PG_DB} OWNER ${PG_USER};\"" || true
    su postgres -c "psql -d ${PG_DB} -c \"CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS postgis; CREATE EXTENSION IF NOT EXISTS postgis_topology;\""

    service postgresql stop

    touch "$SETUP_MARKER"
    echo 'Setup complete.'

fi

# ========================================
# Start services
# ========================================
cd /workspace
set +e

service postgresql start
echo 'PostgreSQL started'

redis-server --daemonize yes
echo 'Redis started'

celery -A celery_app worker --loglevel=info --pool=solo  \
    --queues=analysis.ingest,analysis.voice_text,analysis.segment,analysis.report &
echo 'Celery worker started'

celery -A celery_app flower --port=5555 &
echo 'Flower UI on http://localhost:5555'

fastapi dev api-server.py --host 0.0.0.0
echo 'FastAPI exited.'
exec bash