#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${LLAMA_CPP_DIR:-}" ] && [ -d "$SCRIPT_DIR/llama.cpp" ]; then
  LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
fi

if [ -z "${LLAMA_SERVER_BIN:-}" ]; then
  if [ -n "${LLAMA_CPP_DIR:-}" ]; then
    LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
  else
    LLAMA_SERVER_BIN="llama-server"
  fi
fi

MODEL_PATH="${MODEL_PATH:-${LLAMA_MODEL_PATH:-}}"
HOST="${LLAMA_SERVER_HOST:-127.0.0.1}"
INSTANCES="${LLAMA_SERVER_INSTANCES:-2}"
BASE_PORT="${LLAMA_SERVER_BASE_PORT:-9000}"
PARALLEL="${LLAMA_PARALLEL:-16}"
# Per-session context size; total server ctx_size = CTXSIZE_PER_SESSION * PARALLEL. Use n_predict when unset.
CTXSIZE_PER_SESSION="${LLAMA_CTXSIZE_PER_SESSION:-${LLAMA_N_PREDICT:-2048}}"
LLAMA_SERVER_ARGS="${LLAMA_SERVER_ARGS:-}"

NGINX_BIN="${NGINX_BIN:-nginx}"
NGINX_PORT="${LLAMA_NGINX_PORT:-8088}"

RUN_DIR="${RUN_DIR:-/tmp/llama-rr}"
ENV_STATE_FILE="$RUN_DIR/llama-rr.env"

start() {
  mkdir -p "$RUN_DIR"

  if [ ! -x "$LLAMA_SERVER_BIN" ]; then
    echo "llama-server not found: $LLAMA_SERVER_BIN" >&2
    exit 1
  fi
  if [ -z "$MODEL_PATH" ]; then
    echo "model path not set. Set LLAMA_MODEL_PATH." >&2
    exit 1
  fi
  if [ ! -f "$MODEL_PATH" ]; then
    echo "model not found: $MODEL_PATH" >&2
    exit 1
  fi
  if ! command -v "$NGINX_BIN" >/dev/null 2>&1; then
    echo "nginx not found: $NGINX_BIN" >&2
    exit 1
  fi

  CTX_SIZE=$((CTXSIZE_PER_SESSION * PARALLEL))

  # Parse comma-separated args into array (supports paths with spaces)
  EXTRA_ARGS=()
  if [ -n "$LLAMA_SERVER_ARGS" ]; then
    IFS=',' read -ra EXTRA_ARGS <<< "$LLAMA_SERVER_ARGS"
    # Trim whitespace from each element
    for i in "${!EXTRA_ARGS[@]}"; do
      EXTRA_ARGS[$i]="$(echo "${EXTRA_ARGS[$i]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    done
  fi

  i=0
  while [ "$i" -lt "$INSTANCES" ]; do
    port=$((BASE_PORT + i))
    log="$RUN_DIR/llama-${port}.log"
    "$LLAMA_SERVER_BIN" --host "$HOST" --port "$port" --model "$MODEL_PATH" --parallel "$PARALLEL" --ctx-size "$CTX_SIZE" "${EXTRA_ARGS[@]}" >"$log" 2>&1 &
    echo $! > "$RUN_DIR/llama-${port}.pid"
    i=$((i + 1))
  done

  upstream_lines=""
  i=0
  while [ "$i" -lt "$INSTANCES" ]; do
    port=$((BASE_PORT + i))
    upstream_lines="${upstream_lines}    server ${HOST}:${port};\n"
    i=$((i + 1))
  done

  conf="$RUN_DIR/nginx.conf"
  printf "worker_processes 1;\n" > "$conf"
  printf "pid %s/nginx.pid;\n" "$RUN_DIR" >> "$conf"
  printf "error_log %s/nginx-error.log;\n" "$RUN_DIR" >> "$conf"
  printf "events { worker_connections 1024; }\n" >> "$conf"
  printf "http {\n" >> "$conf"
  printf "  access_log %s/nginx-access.log;\n" "$RUN_DIR" >> "$conf"
  printf "  upstream llama_backend {\n%b  }\n" "$upstream_lines" >> "$conf"
  printf "  server {\n" >> "$conf"
  printf "    listen %s:%s;\n" "$HOST" "$NGINX_PORT" >> "$conf"
  printf "    location / {\n" >> "$conf"
  printf "      proxy_pass http://llama_backend;\n" >> "$conf"
  printf "      proxy_http_version 1.1;\n" >> "$conf"
  printf "      proxy_set_header Connection \"\";\n" >> "$conf"
  printf "    }\n  }\n}\n" >> "$conf"

  "$NGINX_BIN" -c "$conf" -p "$RUN_DIR" -g "daemon off;" >"$RUN_DIR/nginx.stdout" 2>&1 &
  echo $! > "$RUN_DIR/nginx.shell.pid"

  cat > "$ENV_STATE_FILE" <<EOF
HOST=$HOST
INSTANCES=$INSTANCES
BASE_PORT=$BASE_PORT
NGINX_PORT=$NGINX_PORT
EOF

  echo "Started ${INSTANCES} llama-server instances and nginx on http://${HOST}:${NGINX_PORT}"
}

stop() {
  if [ -f "$ENV_STATE_FILE" ]; then
    # shellcheck source=/dev/null
    . "$ENV_STATE_FILE"
  fi

  if [ -f "$RUN_DIR/nginx.pid" ]; then
    kill "$(cat "$RUN_DIR/nginx.pid")" 2>/dev/null || true
  fi
  if [ -f "$RUN_DIR/nginx.shell.pid" ]; then
    kill "$(cat "$RUN_DIR/nginx.shell.pid")" 2>/dev/null || true
  fi
  for pidfile in "$RUN_DIR"/llama-*.pid; do
    [ -f "$pidfile" ] || continue
    kill "$(cat "$pidfile")" 2>/dev/null || true
  done

  if command -v lsof >/dev/null 2>&1; then
    i=0
    while [ "$i" -lt "${INSTANCES:-0}" ]; do
      port=$((BASE_PORT + i))
      pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
      if [ -n "$pid" ]; then
        name=$(ps -p "$pid" -o comm= 2>/dev/null || true)
        case "$name" in
          *llama-server*|*llama-*)
            kill "$pid" 2>/dev/null || true
            ;;
        esac
      fi
      i=$((i + 1))
    done
  fi

  rm -f "$RUN_DIR"/nginx.pid "$RUN_DIR"/nginx.shell.pid "$RUN_DIR"/llama-*.pid "$ENV_STATE_FILE" 2>/dev/null || true
  echo "Stopped."
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  *) echo "Usage: $0 [start|stop]" >&2; exit 1 ;;
esac
