#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint to optionally start an MLflow server and then run training.
# Usage examples:
#   docker run ... image scripts/entrypoint.sh --train-args "--cond_data_dir /data/cond --target_data_dir /data/target"
# Environment variables:
#   START_MLFLOW_SERVER (default: 0) -> if 1, launches local mlflow server
#   MLFLOW_HOST (default: 0.0.0.0)
#   MLFLOW_PORT (default: 5000)
#   MLFLOW_BACKEND_STORE (default: /workspace/mlruns)
#   MLFLOW_ARTIFACT_ROOT (default: /workspace/artifacts)
#   MLFLOW_EXPERIMENT_NAME (default: maskfree_vton)
#   MLFLOW_TRACKING_URI (auto-set if server started locally)
#   TRAIN_SCRIPT (default: train_lora.py)
#   EXTRA_PIP (optional) -> space-separated extra packages to install prior to run
#   PYTHONUNBUFFERED=1 ensures logs flush

START_MLFLOW_SERVER=${START_MLFLOW_SERVER:-0}
MLFLOW_HOST=${MLFLOW_HOST:-0.0.0.0}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_BACKEND_STORE=${MLFLOW_BACKEND_STORE:-/workspace/mlruns}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-/workspace/artifacts}
MLFLOW_EXPERIMENT_NAME=${MLFLOW_EXPERIMENT_NAME:-maskfree_vton}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_lora.py}
EXTRA_PIP=${EXTRA_PIP:-}
USE_ACCELERATE=${USE_ACCELERATE:-0}

# Install extra python packages if provided with EXTRA_PIP
if [ -n "$EXTRA_PIP" ]; then
  echo "[entrypoint] Installing extra pip packages: $EXTRA_PIP"
  pip install --no-cache-dir $EXTRA_PIP
fi

# Optional: start MLflow locally and tell the training script how to find it
if [ "$START_MLFLOW_SERVER" = "1" ]; then
  echo "[entrypoint] Starting MLflow server on $MLFLOW_HOST:$MLFLOW_PORT"
  mkdir -p "$MLFLOW_BACKEND_STORE" "$MLFLOW_ARTIFACT_ROOT"
  mlflow server \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --backend-store-uri "sqlite:///$MLFLOW_BACKEND_STORE/mlflow.db" \
    --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" &
  MLFLOW_PID=$!
  export MLFLOW_TRACKING_URI="http://$MLFLOW_HOST:$MLFLOW_PORT"
  echo "[entrypoint] MLflow server PID $MLFLOW_PID; tracking URI $MLFLOW_TRACKING_URI"
  # Give MLflow a moment to come up
  sleep 3
fi

# Always export the experiment name so Python can pick it up as a fallback
export MLFLOW_EXPERIMENT_NAME

# If first arg is -- just, ignore it
if [ "${1:-}" = "--" ]; then shift; fi

# Collect training arguments
TRAIN_ARGS="$@"

# Run training
if [ "$USE_ACCELERATE" = "1" ]; then
  echo "[entrypoint] Using accelerate launch for: $TRAIN_SCRIPT $TRAIN_ARGS"
  accelerate launch "$TRAIN_SCRIPT" $TRAIN_ARGS
else
  echo "[entrypoint] Running training script: $TRAIN_SCRIPT $TRAIN_ARGS"
  python "$TRAIN_SCRIPT" $TRAIN_ARGS
fi

# Stop mlflow server
if [ "$START_MLFLOW_SERVER" = "1" ]; then
  echo "[entrypoint] Training finished; stopping MLflow server (PID $MLFLOW_PID)"
  kill $MLFLOW_PID || true
fi
