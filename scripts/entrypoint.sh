#!/usr/bin/env bash
set -euo pipefail

# Container entrypoint - starts SSH server and keeps container running
# For services (MLflow, checkpoints): run /app/scripts/start_services.sh
# For training: run /app/scripts/train_lora.sh

echo "[entrypoint] Initializing container..."

# ---------- MAKE CODE PERSISTENT ----------
APP_SRC=/app
APP_RW=/workspace/app

# On first boot, copy the baked /app into the persistent /workspace/app
if [ ! -d "$APP_RW" ] || [ -z "$(ls -A "$APP_RW" 2>/dev/null)" ]; then
  echo "[entrypoint] Seeding $APP_RW from $APP_SRC"
  mkdir -p "$APP_RW"
  cp -a "$APP_SRC"/. "$APP_RW"/
fi

# Prefer the persistent copy for all runtime work
export PYTHONPATH="$APP_RW:$PYTHONPATH"
cd "$APP_RW"

# ---------- SSH SERVER ----------
SSH_USER="${SSH_USER:-trainer}"

# Allow injecting a public key at runtime (set AUTHORIZED_KEY env in RunPod template)
if [[ -n "${AUTHORIZED_KEY:-}" ]]; then
  echo "[entrypoint] Configuring SSH key for user: $SSH_USER"
  mkdir -p "/home/$SSH_USER/.ssh"
  printf '%s\n' "$AUTHORIZED_KEY" >> "/home/$SSH_USER/.ssh/authorized_keys"
  chown -R "$SSH_USER:$SSH_USER" "/home/$SSH_USER/.ssh"
  chmod 700 "/home/$SSH_USER/.ssh"
  chmod 600 "/home/$SSH_USER/.ssh/authorized_keys"
fi

# Generate host keys if missing and start sshd
echo "[entrypoint] Starting SSH server..."
if command -v ssh-keygen >/dev/null 2>&1; then 
  ssh-keygen -A || true
fi
service ssh start >/dev/null 2>&1 || /usr/sbin/sshd || true

# ---------- OPTIONAL EXTRA PACKAGES ----------
EXTRA_PIP=${EXTRA_PIP:-}
if [ -n "$EXTRA_PIP" ]; then
  echo "[entrypoint] Installing extra pip packages: $EXTRA_PIP"
  pip install --no-cache-dir $EXTRA_PIP
fi

# ---------- AUTO-START SERVICES ----------
echo "[entrypoint] Starting services (MLflow + checkpoints)..."
/app/scripts/start_services.sh &
sleep 3

# ---------- HANDLE COMMAND ----------
echo "[entrypoint] Container ready!"
echo ""
echo "================================================"
echo "  SSH Server: Running (port 22)"
echo "  MLflow:     Running (port 5000)"
echo "================================================"
echo ""
echo "Available commands:"
echo "  - Start training:  /app/scripts/train_lora.sh"
echo "  - Manual training: accelerate launch train_lora.py [args]"
echo ""
echo "Services already running:"
echo "  ✓ SSH Server (port 22)"
echo "  ✓ MLflow (http://localhost:5000)"
echo ""

# If arguments provided, execute them (allows running custom commands)
if [ $# -gt 0 ]; then
  echo "[entrypoint] Executing: $@"
  exec "$@"
else
  # Keep container running
  echo "[entrypoint] Container running. Connect via SSH or docker exec."
  tail -f /dev/null
fi
