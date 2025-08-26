#!/bin/bash
set -euo pipefail
# Read username from first argument
DEPLOY_USER="${1:?Usage: $0 <deploy_user>}"
REPO_DIR = "/home/$DEPLOY_USER/projects/homewizard-project"
SERVICE_NAME = "hwe_control_script.service"

# Logging helper
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$REPO_DIR"

# Get the current commit hash
CURRENT_HASH=$(git rev-parse HEAD)

# Fetch latest changes from deploy branch
git fetch origin deploy
git reset --hard origin/deploy

# Get new commit hash
NEW_HASH=$(git rev-parse HEAD)

# Compare hashes
if [ "$CURRENT_HASH" != "$NEW_HASH" ]; then
    log "New changes detected: $CURRENT_HASH → $NEW_HASH."
    log "Updating dependencies and restarting service..."
    poetry install
    sudo systemctl restart "$SERVICE_NAME"
    log "$SERVICE_NAME restarted succesfully"
else
    log "No changes detected. Skipping restart."
fi
