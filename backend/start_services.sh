#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# start_services.sh  —  NeuroPulse | Launch all 4 services
#
# Usage:
#   chmod +x start_services.sh
#   ./start_services.sh
#
# Stops with:  kill $(cat .pids)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pids"

# ── Helpers ────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_port() {
    local port=$1
    if lsof -Pi ":$port" -sTCP:LISTEN -t &>/dev/null; then
        log "ERROR: Port $port is already in use. Aborting."
        exit 1
    fi
}

wait_for_port() {
    local port=$1
    local name=$2
    local max_wait=30
    local elapsed=0
    while ! curl -sf "http://localhost:$port/health" &>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ "$elapsed" -ge "$max_wait" ]; then
            log "WARNING: $name (port $port) did not become healthy within ${max_wait}s"
            return 1
        fi
    done
    log "$name is healthy on port $port"
}

# ── Preflight checks ───────────────────────────────────────────────────────
check_port 8000
check_port 8001
check_port 8002
check_port 8003

# ── Verify virtual environments exist ─────────────────────────────────────
for venv in venv_rppg venv_cnn venv_orch; do
    if [ ! -d "$SCRIPT_DIR/$venv" ]; then
        log "ERROR: $venv not found. Run setup first (see README)."
        exit 1
    fi
done

# ── Clear old PID file ─────────────────────────────────────────────────────
> "$PID_FILE"

# ── Start rPPG service (port 8001) ─────────────────────────────────────────
log "Starting rPPG service on port 8001..."
"$SCRIPT_DIR/venv_rppg/bin/uvicorn" service_rppg:app \
    --host 0.0.0.0 --port 8001 --workers 1 --log-level warning \
    > "$SCRIPT_DIR/logs/rppg.log" 2>&1 &
echo $! >> "$PID_FILE"

# ── Start EfficientNet service (port 8002) ─────────────────────────────────
log "Starting EfficientNet service on port 8002..."
"$SCRIPT_DIR/venv_cnn/bin/uvicorn" service_efficientnet:app \
    --host 0.0.0.0 --port 8002 --workers 1 --log-level warning \
    > "$SCRIPT_DIR/logs/efficientnet.log" 2>&1 &
echo $! >> "$PID_FILE"

# ── Start Swin service (port 8003) ─────────────────────────────────────────
log "Starting Swin service on port 8003..."
"$SCRIPT_DIR/venv_cnn/bin/uvicorn" service_swin:app \
    --host 0.0.0.0 --port 8003 --workers 1 --log-level warning \
    > "$SCRIPT_DIR/logs/swin.log" 2>&1 &
echo $! >> "$PID_FILE"

# ── Wait for sub-services to be healthy ────────────────────────────────────
log "Waiting for sub-services to become healthy..."
sleep 5
wait_for_port 8001 "rPPG"
wait_for_port 8002 "EfficientNet"
wait_for_port 8003 "Swin"

# ── Start Orchestrator (port 8000) ─────────────────────────────────────────
log "Starting Orchestrator on port 8000..."
"$SCRIPT_DIR/venv_orch/bin/uvicorn" orchestrator:app \
    --host 0.0.0.0 --port 8000 --workers 1 --log-level info \
    > "$SCRIPT_DIR/logs/orchestrator.log" 2>&1 &
echo $! >> "$PID_FILE"

log "All services started. PIDs stored in $PID_FILE"
log "  Orchestrator : http://localhost:8000/api/v1/predict"
log "  rPPG         : http://localhost:8001/predict"
log "  EfficientNet : http://localhost:8002/predict"
log "  Swin         : http://localhost:8003/predict"
log ""
log "To stop all:  kill \$(cat $PID_FILE)"
