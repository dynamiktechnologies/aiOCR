#!/usr/bin/env bash
# Start the AI OCR API without Docker (Linux / macOS).
#
#   ./run.sh                            # http://127.0.0.1:8000
#   ./run.sh --port 8080
#   ./run.sh --host 0.0.0.0 --restart   # expose on the LAN, stop a stale instance first
#
# Uses a project-local .venv when present, otherwise python3 on PATH, and only
# installs requirements.txt when imports are missing. Configuration comes from .env.
set -euo pipefail

PORT=8000
HOST=127.0.0.1
RESTART=0

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)    PORT="${2:?--port needs a value}"; shift 2 ;;
    -H|--host)    HOST="${2:?--host needs a value}"; shift 2 ;;
    -r|--restart) RESTART=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"
VENV_PY=".venv/bin/python"

if [[ -x "$VENV_PY" ]]; then
  PY="$VENV_PY"
else
  PY="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PY:-}" ]]; then
  echo "Python 3.10+ not found. Install it (e.g. 'sudo apt install python3 python3-venv') and retry." >&2
  exit 1
fi
echo "Using Python: $PY"

missing="$("$PY" - <<'PY'
import importlib.util
mods = ("fastapi", "uvicorn", "paddleocr", "paddle", "fitz", "docx", "openpyxl",
        "PIL", "openai", "bs4", "dotenv", "numpy")
print(", ".join(m for m in mods if importlib.util.find_spec(m) is None))
PY
)"

if [[ -n "$missing" ]]; then
  echo "Installing dependencies (first run only, this can take a few minutes)..."
  if [[ ! -x "$VENV_PY" ]]; then
    "$PY" -m venv .venv
    PY="$VENV_PY"
    echo "Created .venv -> $PY"
  fi
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r requirements.txt
fi

# --- make sure the port is free ------------------------------------------------
port_busy() {
  "$PY" - "$PORT" <<'PY'
import socket, sys
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(0)          # in use
finally:
    sock.close()
sys.exit(1)              # free
PY
}

if port_busy; then
  if [[ "$RESTART" == "1" ]] && command -v lsof >/dev/null 2>&1; then
    stale="$(lsof -ti "tcp:$PORT" | tr '\n' ' ' || true)"
    if [[ -n "${stale// /}" ]]; then
      echo "Stopping process(es) holding port $PORT: $stale"
      # shellcheck disable=SC2086
      kill $stale 2>/dev/null || true
      sleep 2
    fi
  fi
  if port_busy; then
    owner="$(command -v lsof >/dev/null 2>&1 && lsof -ti "tcp:$PORT" | head -n1 || echo unknown)"
    echo "Port $PORT is already in use (PID: $owner). Stop it, use --port, or re-run with --restart." >&2
    exit 1
  fi
fi

# --- advisory checks ------------------------------------------------------------
if [[ -f .env ]]; then
  if grep -Eq '^[[:space:]]*OCR_DEMO_MODE[[:space:]]*=[[:space:]]*(1|true|yes|on)[[:space:]]*$' .env; then
    echo "WARNING: OCR_DEMO_MODE is on - the portal offers one-click admin login. Set it to false before exposing this server." >&2
  fi
  if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]] &&
     ! grep -Eq '^[[:space:]]*OCR_COOKIE_SECURE[[:space:]]*=[[:space:]]*(1|true|yes|on)[[:space:]]*$' .env; then
    echo "WARNING: bound to $HOST without OCR_COOKIE_SECURE=true - session cookies would travel over plain HTTP." >&2
  fi
fi

shown="$HOST"
[[ "$HOST" == "0.0.0.0" ]] && shown="localhost"
printf '\nAI OCR listening on http://%s:%s\n' "$shown" "$PORT"
printf '  portal   http://%s:%s/\n  demo     http://%s:%s/demo\n  API docs http://%s:%s/docs\n' \
  "$shown" "$PORT" "$shown" "$PORT" "$shown" "$PORT"
printf 'Press Ctrl+C to stop.\n\n'

exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
