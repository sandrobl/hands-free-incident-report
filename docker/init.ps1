# SAM3 Docker Setup Script for Windows 10 + GPU
# Single file solution - Double-click to run

# Resolve project root (parent of docker/)
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# Load project-root .env file if present
$envFile = Join-Path $projectRoot '.env'
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
        }
    }
}
$markerFile = Join-Path $projectRoot '.sam3_setup_done'
if (Test-Path $markerFile) { Remove-Item $markerFile }

Write-Host '========================================' -ForegroundColor Cyan
Write-Host 'SAM3 Docker Setup with GPU Support' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host ''

# Check if Docker is running
Write-Host 'Checking if Docker is running...' -ForegroundColor Yellow
$dockerRunning = docker info 2>$null
if (-not $dockerRunning) {
    Write-Host 'ERROR: Docker is not running!' -ForegroundColor Red
    Write-Host 'Please start Docker Desktop and run this script again.' -ForegroundColor Red
    Write-Host ''
    Read-Host 'Press Enter to exit'
    exit 1
}
Write-Host 'Docker is running ✓' -ForegroundColor Green
Write-Host ''

# Check GPU availability
Write-Host 'Checking GPU availability...' -ForegroundColor Yellow
$gpuTest = docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host 'WARNING: GPU test failed!' -ForegroundColor Red
    Write-Host 'Make sure:' -ForegroundColor Yellow
    Write-Host '  1. Nvidia drivers are installed' -ForegroundColor Yellow
    Write-Host '  2. Docker Desktop has GPU support enabled' -ForegroundColor Yellow
    Write-Host ''
    $continue = Read-Host 'Continue anyway? (y/n)'
    if ($continue -ne 'y') {
        exit 1
    }
} else {
    Write-Host 'GPU detected ✓' -ForegroundColor Green
    Write-Host ''
}

# Pull the CUDA image
Write-Host 'Pulling CUDA Docker image (this may take a few minutes)...' -ForegroundColor Yellow
docker pull nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04
if ($LASTEXITCODE -ne 0) {
    Write-Host 'ERROR: Failed to pull Docker image!' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}
Write-Host 'Image pulled successfully ✓' -ForegroundColor Green
Write-Host ''

Write-Host 'Workspace directory: ' -NoNewline
Write-Host $projectRoot -ForegroundColor Cyan
Write-Host ''

Write-Host 'Starting Docker container...' -ForegroundColor Yellow
Write-Host ''
Write-Host '========================================' -ForegroundColor Green
Write-Host 'Container is starting!' -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Green
Write-Host ''

Write-Host 'Starting Caddy reverse proxy...' -ForegroundColor Yellow
docker-compose -f (Join-Path $PSScriptRoot 'docker-compose.yml') up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host 'ERROR: Failed to start Caddy!' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}
Write-Host 'Caddy started ✓' -ForegroundColor Green
Write-Host ''

# Forward only the simple single-line vars the setup script needs
$envArgs = @()
foreach ($pair in @(
    @{ Name = 'HUGGING_FACE_HUB_TOKEN'; Value = $env:HUGGING_FACE_HUB_TOKEN },
    @{ Name = 'POSTGRES_USER'; Value = $env:POSTGRES_USER },
    @{ Name = 'POSTGRES_PASSWORD'; Value = $env:POSTGRES_PASSWORD },
    @{ Name = 'POSTGRES_DB'; Value = $env:POSTGRES_DB },
    @{ Name = 'DATABASE_URL'; Value = $env:DATABASE_URL }
)) {
    if ($pair.Value) {
        $envArgs += '-e', ("{0}={1}" -f $pair.Name, $pair.Value)
    }
}

# Use repo setup.sh directly (avoid temp file mount issues)
$setupPath = Join-Path $PSScriptRoot 'setup.sh'
if (-not (Test-Path $setupPath)) {
    Write-Host 'ERROR: setup.sh not found in the docker directory.' -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

# Ensure local uploads directory exists
$uploadsDir = Join-Path $projectRoot 'uploads'
if (-not (Test-Path $uploadsDir)) { New-Item -ItemType Directory -Path $uploadsDir | Out-Null }

# Run container
docker run --gpus all -it -p 8000:8000 -p 5555:5555 -p 5432:5432 -v "${projectRoot}:/workspace" -v "${uploadsDir}:/data/uploads" -v "${setupPath}:/setup.sh" @envArgs --name sam3-container nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04 bash /setup.sh

Write-Host ''
Write-Host 'Container exited. Goodbye!' -ForegroundColor Cyan
Read-Host 'Press Enter to close'