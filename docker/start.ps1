docker-compose -f (Join-Path $PSScriptRoot 'docker-compose.yml') up -d
docker start -ai sam3-container