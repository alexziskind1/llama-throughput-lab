# Llama.cpp Throughput Launcher

https://www.youtube.com/watch?v=L9QZ97y9Exg

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

- `LLAMA_SERVER_ARGS`: extra args passed to `llama-server` using **comma-separated** format with `=` for values (e.g. `--ctx-size=4096,-fa=1,--mmproj=/path/to/model.bin`). Can be set via the **Advanced Args** menu in the launcher or via Env Overrides. If `--ctx-size` or `--parallel` appear in these args, the corresponding computed value is **skipped entirely** (not duplicated). See [Advanced Server Arguments](#advanced-server-arguments) for details.
- `LLAMA_CTXSIZE_PER_SESSION`: context size per session (tokens). If set (e.g. via env overrides), the server is started with `--ctx-size (ctxsizePerSession * parallel)`. If not set, context is derived from `LLAMA_N_PREDICT`. Formula: `ctx_size = (LLAMA_CTXSIZE_PER_SESSION or LLAMA_N_PREDICT) * LLAMA_PARALLEL`. The dialog UI does not expose this; it always uses the single-test token value for context.
- `LLAMA_SERVER_HOST`: host for llama-server (default `127.0.0.1`).
- `LLAMA_SERVER_PORT`: fixed port for single-server tests (optional).
- `LLAMA_SERVER_INSTANCES`: number of servers for round-robin tests/sweeps.
- `LLAMA_SERVER_BASE_PORT`: base port for multi-server runs (default `9000`).
- `LLAMA_NGINX_PORT`: nginx listen port (default `8088`).
- `LLAMA_READY_TIMEOUT`: seconds to wait for model readiness.
- `LLAMA_SERVER_BIND_TIMEOUT`: seconds to wait for server to bind (default 180; increase if model load is slow).
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

- `LLAMA_MAX_TOKENS_LIST`: list of max tokens (round-robin sweep). Values ≤2048 use one server run (ctx=2048×parallel); values >2048 restart the server per value (ctx=n_predict×parallel).
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

## Advanced Server Arguments

The launcher exposes an **Advanced Args** field (main menu option 6, and in the
round-robin submenu) that lets you pass arbitrary flags directly to `llama-server`
via the `LLAMA_SERVER_ARGS` environment variable.

### Format

Use **comma-separated** tokens. Join flags and values with `=`:

```
--ctx-size=4096,-fa=1,--mmproj=/path with spaces/file.bin
```

Each comma-delimited token becomes one argument. Paths with spaces are fully
supported because commas (not spaces) delimit arguments.

### Override behavior

If `--ctx-size` or `--parallel` are present in `LLAMA_SERVER_ARGS`, the launcher
**skips injecting** the corresponding computed value entirely — it does not pass
the flag twice and rely on last-value-wins. This matters because `llama-server`
divides the total `--ctx-size` budget among `--parallel` slots. If you set
`--ctx-size=262144,--parallel=16`, each slot gets 16 384 tokens of context.

For example, if you enter `--ctx-size=262144` in Advanced Args, the launcher will
not inject its own computed `--ctx-size` at all, so the value you set is the one
the server sees.

### Precedence: Env Overrides vs Advanced Args

If you set `LLAMA_SERVER_ARGS` in the **Env Overrides** field (e.g.
`LLAMA_SERVER_ARGS="--ctx-size=8192"`), it takes priority over the Advanced Args
field. The Advanced Args value is only used when `LLAMA_SERVER_ARGS` is not
already present in Env Overrides.

### Sweep interaction

The full sweep (`scripts/full_sweep.py`) filters `--parallel`, `--batch-size`,
and `--ubatch` from `LLAMA_SERVER_ARGS` and replaces them with sweep-specific
values. Other arguments (e.g. `-fa 1`, `--mmproj`) are preserved.

### Limitations

- **Reserved flags in sweeps**: `--parallel`, `--batch-size`, and `--ubatch` are
  auto-managed by sweep scripts and will be stripped/replaced. Do not rely on
  setting these through Advanced Args when running sweeps.

## Examples

### Launcher

Run the launcher and pass overrides in the dialog:
```
LLAMA_CONCURRENCY=64 LLAMA_NUM_REQUESTS=64 LLAMA_SERVER_ARGS="--parallel=64"
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



## Analyze the Data

```bash
python analyze-data.py --file results/full_sweep/full_sweep_20260131_150913.csv --field errors --order desc --count 10
```

Parameters:
```plaintext
--file    ... which file you want to process (required)
--field   ... which field do you want to sort by (throughput_tps is the default if none is given)
--order   ... 'asc' or 'desc' for ascending or descending (descending is the default if not given)
--count   ... how many records to show (5 is the default)
```

Output will look something like this:

```plaintext
$ python analyze-data.py --file results/full_sweep/full_sweep_20260131_150913.csv
instances | parallel | batch   | ubatch  | concurrency | throughput_tps | total_tokens | elapsed_s | errors
-----------------------------------------------------------------------------------------------------------
      2.0 |     64.0 | default | default |       128.0 |          359.4 |      16384.0 |     45.59 |    0.0
      2.0 |     32.0 | default | default |        64.0 |          217.6 |       8192.0 |     37.64 |    0.0
      2.0 |     64.0 | default | default |        64.0 |          217.5 |       8192.0 |     37.67 |    0.0
      2.0 |     32.0 | default | default |       128.0 |           91.0 |       8448.0 |     92.82 |   62.0
      2.0 |     64.0 | default | default |        32.0 |           75.0 |       4096.0 |     54.59 |    0.0


$ python analyze-data.py --file results/full_sweep/full_sweep_20260131_150913.csv --field errors --order desc --count 10
instances | parallel | batch   | ubatch  | concurrency | throughput_tps | total_tokens | elapsed_s | errors
-----------------------------------------------------------------------------------------------------------
      4.0 |     32.0 | default | default |       128.0 |            0.0 |          0.0 |     14.07 |  128.0
      4.0 |     16.0 | default | default |       128.0 |           19.1 |       2560.0 |    134.05 |  108.0
      2.0 |     16.0 | default | default |       128.0 |           46.7 |       4352.0 |     93.12 |   94.0
      4.0 |     32.0 | default | default |        64.0 |            0.0 |          0.0 |     14.05 |   64.0
      2.0 |     32.0 | default | default |       128.0 |           91.0 |       8448.0 |     92.82 |   62.0
      4.0 |     16.0 | default | default |        64.0 |           16.2 |       2176.0 |    134.08 |   47.0
      4.0 |     32.0 | default | default |        32.0 |            0.0 |          0.0 |     28.94 |   32.0
      2.0 |     16.0 | default | default |        64.0 |           48.6 |       4352.0 |      89.6 |   30.0
      2.0 |     16.0 | default | default |        32.0 |           73.9 |       4096.0 |     55.46 |    0.0
      2.0 |     32.0 | default | default |        32.0 |           73.6 |       4096.0 |     55.64 |    0.0
```
