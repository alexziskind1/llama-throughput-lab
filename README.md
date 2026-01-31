# Llama.cpp Throughput Launcher

This repo provides a dialog-based launcher to run llama.cpp throughput tests and sweeps.

## Quick Start

1) Create and activate a virtual environment (optional):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install `dialog`:
- macOS: `brew install dialog`
- Debian/Ubuntu: `sudo apt-get install dialog`
- Fedora: `sudo dnf install dialog`
- Arch: `sudo pacman -S dialog`

3) Run the launcher:
```bash
./run_llama_tests.py
```

The launcher lets you pick a test/sweep, select a GGUF model file, and enter env overrides.

## Run With Launcher

Use the interactive menu to pick tests or sweeps and supply optional env overrides.
The launcher will try to auto-detect a `.gguf` in common locations unless you
pick one or set `LLAMA_MODEL_PATH`.
It also auto-detects `llama-server` from `LLAMA_CPP_DIR`, walking up parent
directories to find `llama.cpp`, or `PATH` unless you set it in the menu.

Model auto-detection searches common locations including `./models`, `./llama.cpp/models`,
`/models`, `~/models`, `~/Models`, `~/Downloads`, and `~/.cache/lm-studio/models`.

If you need to access the server from another machine, set `LLAMA_SERVER_HOST=0.0.0.0`
so nginx and llama-server bind to all interfaces (default is `127.0.0.1`).

```bash
./run_llama_tests.py
```

## Run Directly (No Launcher)

Run any test or sweep directly with Python and environment variables.

```bash
.venv/bin/python -m unittest tests/test_llama_server_single.py
.venv/bin/python -m unittest tests/test_llama_server_concurrent.py
.venv/bin/python -m unittest tests/test_llama_server_round_robin.py
.venv/bin/python tests/test_llama_server_threads_sweep.py
.venv/bin/python scripts/round_robin_sweep.py
.venv/bin/python scripts/full_sweep.py
```

## Launcher Options

Tests are quick, pass/fail checks you can run like normal unit tests. Sweeps are longer
benchmark runs that explore parameter ranges and report the best throughput.

Tests:
- Single request
- Concurrent requests
- Round-robin (nginx + multiple servers, requires `nginx`)

Sweeps:
- Threads (--threads/--threads-http)
- Round-robin (max_tokens x concurrency, requires `nginx`)
- Full (instances x parallel x concurrency, requires `nginx`)

Utilities:
- Configure and run round robin (submenu to set instances/ports/parallel, then start/stop)

These use the same env overrides (e.g., `LLAMA_SERVER_INSTANCES`, `LLAMA_PARALLEL`,
`LLAMA_SERVER_BASE_PORT`, `LLAMA_NGINX_PORT`, `LLAMA_SERVER_HOST`, `LLAMA_MODEL_PATH`,
`LLAMA_SERVER_BIN`).

## Requirements

- llama.cpp built with `llama-server` available.
- `nginx` installed for round-robin tests/sweeps (`brew install nginx` on macOS).
- Model in GGUF format.

You must provide a GGUF model path via the launcher or `LLAMA_MODEL_PATH`.

## llama.cpp Prerequisite

You need a local build of llama.cpp with the `llama-server` binary available.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
```

By default this repo will look for `llama-server` in `./llama.cpp` (a sibling of this
repo). If your llama.cpp lives elsewhere, set one of:

```
LLAMA_CPP_DIR=/path/to/llama.cpp
LLAMA_SERVER_BIN=/path/to/llama-server
```

## Environment Variables

You can supply overrides in the launcher (space-separated `KEY=VALUE` pairs), or set them in your shell.

### Core Paths

- `LLAMA_MODEL_PATH`: GGUF model path.
- `LLAMA_MODEL_DIRS`: colon- or comma-separated directories to auto-detect a `.gguf` (launcher only).
- `LLAMA_MODEL_SEARCH_DEPTH`: max directory depth to scan when auto-detecting (default `4`).
- `LLAMA_CPP_DIR`: llama.cpp repo path.
- `LLAMA_SERVER_BIN`: path to `llama-server` binary.

### Server Behavior

- `LLAMA_SERVER_ARGS`: extra args passed to `llama-server` (e.g. `--parallel 64`).
- `LLAMA_SERVER_HOST`: host for llama-server (default `127.0.0.1`).
- `LLAMA_SERVER_PORT`: fixed port for single-server tests (optional).
- `LLAMA_SERVER_INSTANCES`: number of servers for round-robin tests/sweeps.
- `LLAMA_SERVER_BASE_PORT`: base port for multi-server runs (default `9000`).
- `LLAMA_NGINX_PORT`: nginx listen port (default `8088`).
- `LLAMA_READY_TIMEOUT`: seconds to wait for model readiness.
- `LLAMA_STARTUP_DELAY_S`: delay between starting servers (stagger startup).

### Request Controls

- `LLAMA_PROMPT`: prompt text.
- `LLAMA_N_PREDICT`: tokens to generate per request.
- `LLAMA_TEMPERATURE`: sampling temperature.
- `LLAMA_CONCURRENCY`: concurrent requests (tests).
- `LLAMA_NUM_REQUESTS`: total requests per run (tests/sweeps).

### Threads Sweeps

- `LLAMA_THREADS_LIST`: comma/space list for `--threads`.
- `LLAMA_THREADS_HTTP_LIST`: list for `--threads-http` (use `default` for unset).
- `LLAMA_THREADS_HTTP`: single value override for `--threads-http` (legacy).

### Sweep Controls

- `LLAMA_MAX_TOKENS_LIST`: list of max tokens (round-robin sweep).
- `LLAMA_CONCURRENCY_LIST`: list of concurrencies (sweeps).
- `LLAMA_INSTANCES_LIST`: list of instance counts (full sweep).
- `LLAMA_PARALLEL_LIST`: list of `--parallel` values (full sweep).
- `LLAMA_BATCH_LIST`: list for `--batch-size` (round-robin/full sweep, use `default` to skip).
- `LLAMA_UBATCH_LIST`: list for `--ubatch` (round-robin/full sweep, use `default` to skip).
- `LLAMA_REQUESTS_MULTIPLIER`: if `LLAMA_NUM_REQUESTS` is unset, total requests = concurrency * multiplier.
- `LLAMA_CONTINUE_ON_ERROR`: set to `0` to stop on the first failing config (default continues).
- `LLAMA_REQUEST_TIMEOUT`: per-request timeout (seconds).
- `LLAMA_RETRY_ATTEMPTS`: retries for transient HTTP errors.
- `LLAMA_RETRY_SLEEP_S`: base retry backoff (seconds).
- `LLAMA_CELL_PAUSE_S`: pause between sweep cells (seconds).
- `LLAMA_WARMUP_REQUESTS`: warmup requests before a sweep run.
- `LLAMA_RESULTS_DIR`: base directory for sweep output files (default `results`).

## Examples

### Launcher

Run the launcher and pass overrides in the dialog:
```
LLAMA_CONCURRENCY=64 LLAMA_NUM_REQUESTS=64 LLAMA_SERVER_ARGS="--parallel 64"
```

Note: `LLAMA_SERVER_ARGS` is for fixed runs. For the full sweep, use
`LLAMA_PARALLEL_LIST` (it already sweeps `--parallel`), so you can omit
`LLAMA_SERVER_ARGS`.

### Direct

Run a concurrent test with custom concurrency and requests:
```bash
LLAMA_CONCURRENCY=64 LLAMA_NUM_REQUESTS=64 \
.venv/bin/python -m unittest tests/test_llama_server_concurrent.py
```

Run full sweep with custom ranges:
```bash
LLAMA_INSTANCES_LIST="2,4,8,16" \
LLAMA_PARALLEL_LIST="16,32,64" \
LLAMA_CONCURRENCY_LIST="32,64,128" \
.venv/bin/python scripts/full_sweep.py
```

Sweep results are written incrementally to:
```
results/full_sweep/full_sweep_<timestamp>.csv
results/round_robin_sweep/round_robin_sweep_<timestamp>.csv
```
Progress updates are printed to stderr during sweeps (completed/total and elapsed time).
