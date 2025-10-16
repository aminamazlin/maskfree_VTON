#!/usr/bin/env bash
set -euo pipefail

# Start MLflow and download model checkpoints
# This script is separate from training so you can start services independently

MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_BACKEND_STORE=${MLFLOW_BACKEND_STORE:-/workspace/app/mlflow/mlruns}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-/workspace/app/mlflow/artifacts}
MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-maskfree_vton}

# How to run MLflow: "local" (pip-installed mlflow CLI) or "docker" (containerized)
MLFLOW_RUN_MODE=${MLFLOW_RUN_MODE:-local}

# Docker-related settings (only used when MLFLOW_RUN_MODE=docker)
MLFLOW_DOCKERFILE=${MLFLOW_DOCKERFILE:-/workspace/app/mlflow/Dockerfile}
MLFLOW_IMAGE=${MLFLOW_IMAGE:-mlflow-server:latest}
MLFLOW_BUILD=${MLFLOW_BUILD:-0} # set to 1 to build image
MLFLOW_CONTAINER_NAME=${MLFLOW_CONTAINER_NAME:-mlflow_server}
MLFLOW_DOCKER_RUN_EXTRA_ARGS=${MLFLOW_DOCKER_RUN_EXTRA_ARGS:-}

# --------- Checkpoint auto-download ---------
# Default to the repo path in this workspace
GSA_CHECKPOINT_DIR=${GSA_CHECKPOINT_DIR:-/workspace/app/utils/Grounded-Segment-Anything/checkpoints}
GDINO_WEIGHTS_URL=${GDINO_WEIGHTS_URL:-https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth}
SAM_WEIGHTS_URL=${SAM_WEIGHTS_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth}
SKIP_CHECKPOINT_DOWNLOAD=${SKIP_CHECKPOINT_DOWNLOAD:-0}

echo "[services] Starting MLflow and downloading model checkpoints..."

# Download model checkpoints if missing
if [ "$SKIP_CHECKPOINT_DOWNLOAD" = "1" ]; then
  echo "[services] SKIP_CHECKPOINT_DOWNLOAD=1, skipping checkpoint downloads"
else
  mkdir -p "$GSA_CHECKPOINT_DIR"
  
  # GroundingDINO
  if [ ! -f "$GSA_CHECKPOINT_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "[services] Downloading GroundingDINO weights -> $GSA_CHECKPOINT_DIR/groundingdino_swint_ogc.pth"
    if ! wget -q --show-progress -O "$GSA_CHECKPOINT_DIR/groundingdino_swint_ogc.pth" "$GDINO_WEIGHTS_URL"; then
      echo "[services][WARN] Failed to download GroundingDINO weights from $GDINO_WEIGHTS_URL" >&2
    fi
  else
    echo "[services] Found GroundingDINO weights (skip)"
  fi
  
  # SAM
  if [ ! -f "$GSA_CHECKPOINT_DIR/sam_vit_h_4b8939.pth" ]; then
    echo "[services] Downloading SAM (ViT-H) weights -> $GSA_CHECKPOINT_DIR/sam_vit_h_4b8939.pth"
    if ! wget -q --show-progress -O "$GSA_CHECKPOINT_DIR/sam_vit_h_4b8939.pth" "$SAM_WEIGHTS_URL"; then
      echo "[services][WARN] Failed to download SAM weights from $SAM_WEIGHTS_URL" >&2
    fi
  else
    echo "[services] Found SAM weights (skip)"
  fi
fi

# Start MLflow server (local or docker)
echo "[services] Starting MLflow server ($MLFLOW_RUN_MODE) on $MLFLOW_HOST:$MLFLOW_PORT"
mkdir -p "$MLFLOW_BACKEND_STORE" "$MLFLOW_ARTIFACT_ROOT"

if [ "$MLFLOW_RUN_MODE" = "docker" ]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[services][ERROR] Docker is not available. Install/enable Docker or set MLFLOW_RUN_MODE=local." >&2
    exit 1
  fi

  # Build image if requested or missing
  if [ "$MLFLOW_BUILD" = "1" ] || ! docker image inspect "$MLFLOW_IMAGE" >/dev/null 2>&1; then
    if [ ! -f "$MLFLOW_DOCKERFILE" ]; then
      echo "[services][ERROR] Dockerfile not found at $MLFLOW_DOCKERFILE" >&2
      exit 1
    fi
    echo "[services] Building MLflow image $MLFLOW_IMAGE from $MLFLOW_DOCKERFILE"
    if ! docker build -t "$MLFLOW_IMAGE" -f "$MLFLOW_DOCKERFILE" "$(dirname "$MLFLOW_DOCKERFILE")"; then
      echo "[services][ERROR] Docker build failed" >&2
      exit 1
    fi
  else
    echo "[services] Using existing MLflow image: $MLFLOW_IMAGE"
  fi

  # Clean up existing container with the same name
  if docker ps -a --format '{{.Names}}' | grep -wq "$MLFLOW_CONTAINER_NAME"; then
    echo "[services] Removing existing container $MLFLOW_CONTAINER_NAME"
    docker rm -f "$MLFLOW_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi

  echo "[services] Launching MLflow container: $MLFLOW_CONTAINER_NAME"
  CONTAINER_ID=$(docker run -d \
    --name "$MLFLOW_CONTAINER_NAME" \
    -p "$MLFLOW_PORT:$MLFLOW_PORT" \
    -v "$MLFLOW_BACKEND_STORE":/mlflow/backend \
    -v "$MLFLOW_ARTIFACT_ROOT":/mlflow/artifacts \
    $MLFLOW_DOCKER_RUN_EXTRA_ARGS \
    "$MLFLOW_IMAGE" \
    mlflow server \
      --host 0.0.0.0 \
      --port "$MLFLOW_PORT" \
      --backend-store-uri "sqlite:////mlflow/backend/mlflow.db" \
      --default-artifact-root "/mlflow/artifacts")

  if [ -z "${CONTAINER_ID:-}" ]; then
    echo "[services][ERROR] Failed to start MLflow container" >&2
    exit 1
  fi

  export MLFLOW_TRACKING_URI="http://$MLFLOW_HOST:$MLFLOW_PORT"
  export MLFLOW_EXPERIMENT_NAME
  echo "[services] MLflow container started (ID: $CONTAINER_ID, name: $MLFLOW_CONTAINER_NAME)"
  echo "[services] MLflow tracking URI: $MLFLOW_TRACKING_URI"
  echo "[services] MLflow experiment: $MLFLOW_EXPERIMENT_NAME"
  echo "[services] Services ready!"

  # Ensure the container is stopped when this script exits
  cleanup() {
    echo "[services] Stopping MLflow container ($MLFLOW_CONTAINER_NAME)"
    docker stop -t 3 "$MLFLOW_CONTAINER_NAME" >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  # Block until container exits
  docker wait "$CONTAINER_ID" >/dev/null 2>&1 || true
else
  if ! command -v mlflow >/dev/null 2>&1; then
    echo "[services][ERROR] 'mlflow' CLI not found. Install it (e.g., 'pip install -r /workspace/app/mlflow/requirements.txt' or set MLFLOW_RUN_MODE=docker)." >&2
    exit 1
  fi

  # Allow external access by default in cloud/container environments
  MLFLOW_ALLOWED_HOSTS=${MLFLOW_ALLOWED_HOSTS:-'*'}
  
  # Set a secret key for CSRF protection if using basic-auth (or disable auth)
  # For development, you can disable auth by removing --app-name basic-auth
  export MLFLOW_FLASK_SERVER_SECRET_KEY=${MLFLOW_FLASK_SERVER_SECRET_KEY:-'dev-secret-key-change-in-production'}
  
  mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "sqlite:///$MLFLOW_BACKEND_STORE/mlflow.db" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" \
    --allowed-hosts "$MLFLOW_ALLOWED_HOSTS" &

  MLFLOW_PID=$!
  export MLFLOW_TRACKING_URI="http://$MLFLOW_HOST:$MLFLOW_PORT"
  export MLFLOW_EXPERIMENT_NAME

  echo "[services] MLflow server started (PID: $MLFLOW_PID)"
  echo "[services] MLflow tracking URI: $MLFLOW_TRACKING_URI"
  echo "[services] MLflow experiment: $MLFLOW_EXPERIMENT_NAME"
  echo "[services] Services ready!"

  # Keep this script running to maintain the background process
  wait $MLFLOW_PID
fi
