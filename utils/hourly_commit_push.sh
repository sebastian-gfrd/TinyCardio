#!/usr/bin/env bash
# ==============================================================================
# TinyCardio Hourly Commit & Push Automation
# ==============================================================================
set -e

STAGE_NAME="$1"
COMMIT_MSG="$2"

if [ -z "$STAGE_NAME" ] || [ -z "$COMMIT_MSG" ]; then
    echo "Usage: ./utils/hourly_commit_push.sh <stage_tag> <commit_message>"
    echo "Example: ./utils/hourly_commit_push.sh 'h1-pipeline' 'feat(pipeline): Ingestion and resampling'"
    exit 1
fi

echo "=== Staging changes ==="
git add .

if git diff-index --quiet HEAD --; then
    echo "No new changes to commit."
else
    echo "=== Committing: [$STAGE_NAME] $COMMIT_MSG ==="
    git commit -m "[$STAGE_NAME] $COMMIT_MSG"
fi

if git remote get-url origin > /dev/null 2>&1; then
    echo "=== Pushing to remote repository ==="
    git push origin main
    echo "Push successful!"
else
    echo "Note: Remote 'origin' is not configured yet."
    echo "To link your GitHub/GitLab repository, run:"
    echo "  git remote add origin https://github.com/<your-user>/<your-repo>.git"
    echo "  git push -u origin main"
fi
