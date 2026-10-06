#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv-reranker/bin/streamlit" run "$PROJECT_ROOT/Streamlit/app.py" \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false
