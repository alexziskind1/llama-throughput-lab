import csv
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed .*socket")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.llama_server_test_utils import (
    extract_token_count,
    parse_comma_args,
    post_json,
    start_llama_servers,
    start_nginx_round_robin,
)


def _parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


def _parse_optional_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    result = []
    for item in parts:
        if not item:
            continue
        if item.lower() == "default":
            result.append(None)
        else:
            result.append(int(item))
    return result or [None]


def init_results_file(subdir, prefix):
    base_dir = Path(os.environ.get("LLAMA_RESULTS_DIR", "results")).expanduser()
    results_dir = base_dir / subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return results_dir / f"{prefix}_{timestamp}.csv"


def _format_cell(value, width):
    if value is None:
        return " " * width
    return f"{value:.1f}".rjust(width)


def post_json_with_retry(url, payload, max_attempts=8, base_sleep_s=0.5):
    for attempt in range(max_attempts):
        try:
            return post_json(url, payload)
        except RuntimeError as exc:
            message = str(exc)
            retryable = any(
                code in message
                for code in (
                    "HTTP error 500",
                    "HTTP error 502",
                    "HTTP error 503",
                    "HTTP error 504",
                    "Loading model",
                )
            )
            if retryable:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(base_sleep_s * (attempt + 1))
                continue
            raise


def run_batch(base_url, prompt, n_predict, concurrency, total_requests, temperature):
    start_time = time.time()
    results = []
    errors = 0
    last_error = None
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_json_with_retry,
                f"{base_url}/completion",
                {
                    "prompt": prompt,
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "stream": False,
                },
            )
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors += 1
                last_error = exc
    total_time = time.time() - start_time

    total_tokens = sum(extract_token_count(result) for result in results)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    return {
        "throughput": throughput,
        "total_tokens": total_tokens,
        "elapsed": total_time,
        "errors": errors,
        "last_error": last_error,
    }


def _build_server_args(base_args, batch_size, ubatch_size):
    cleaned = []
    skip_next = False
    for arg in base_args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--batch-size", "--ubatch", "-b"}:
            skip_next = True
            continue
        if arg.startswith("--batch-size="):
            continue
        if arg.startswith("--ubatch="):
            continue
        cleaned.append(arg)

    if batch_size is not None:
        cleaned += ["--batch-size", str(batch_size)]
    if ubatch_size is not None:
        cleaned += ["--ubatch", str(ubatch_size)]
    return cleaned


def main():
    prompt = os.environ.get(
        "LLAMA_PROMPT",
        "Share three optimization tips for model serving.",
    )
    temperature = float(os.environ.get("LLAMA_TEMPERATURE", "0.3"))
    instance_count = int(os.environ.get("LLAMA_SERVER_INSTANCES", "2"))
    base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
    nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
    ready_timeout_s = int(os.environ.get("LLAMA_READY_TIMEOUT", "180"))
    startup_delay_s = float(os.environ.get("LLAMA_STARTUP_DELAY_S", "0.0"))
    base_args = parse_comma_args(os.environ.get("LLAMA_SERVER_ARGS", ""))

    # n_predict ≤ threshold: one server run with ctx = 2048 * parallel. n_predict > threshold: restart per value.
    CTXSIZE_THRESHOLD = 2048
    max_tokens_list = _parse_int_list(
        os.environ.get("LLAMA_MAX_TOKENS_LIST"),
        "128,256,512,1024",
    )
    max_tokens_low = [m for m in max_tokens_list if m <= CTXSIZE_THRESHOLD]
    max_tokens_high = [m for m in max_tokens_list if m > CTXSIZE_THRESHOLD]
    parallel = int(os.environ.get("LLAMA_PARALLEL", "1"))

    batch_list = _parse_optional_int_list(
        os.environ.get("LLAMA_BATCH_LIST"),
        "default",
    )
    ubatch_list = _parse_optional_int_list(
        os.environ.get("LLAMA_UBATCH_LIST"),
        "default",
    )
    concurrency_list = _parse_int_list(
        os.environ.get("LLAMA_CONCURRENCY_LIST"),
        "1,2,4,8,16,32,64,128,256,512,1024",
    )

    total_requests_env = os.environ.get("LLAMA_NUM_REQUESTS")
    requests_multiplier = int(os.environ.get("LLAMA_REQUESTS_MULTIPLIER", "1"))
    cell_pause_s = float(os.environ.get("LLAMA_CELL_PAUSE_S", "0.0"))
    continue_on_error = os.environ.get("LLAMA_CONTINUE_ON_ERROR", "1").lower() not in {
        "0",
        "false",
        "no",
    }

    warmup_requests = int(os.environ.get("LLAMA_WARMUP_REQUESTS", "2"))

    if instance_count < 1:
        instance_count = 1
    if requests_multiplier < 1:
        requests_multiplier = 1

    results_path = init_results_file("round_robin_sweep", "round_robin_sweep")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_file = results_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(results_file)
    writer.writerow(
        [
            "batch",
            "ubatch",
            "max_tokens",
            "concurrency",
            "throughput_tps",
            "total_tokens",
            "elapsed_s",
            "errors",
        ]
    )
    results_file.flush()

    print(f"results_file={results_path}")

    total_runs = len(batch_list) * len(ubatch_list) * len(max_tokens_list) * len(
        concurrency_list
    )
    completed = 0
    sweep_start = time.time()

    def record_row(
        batch_label,
        ubatch_label,
        max_tokens,
        concurrency,
        throughput,
        total_tokens,
        elapsed,
        errors,
    ):
        nonlocal completed
        writer.writerow(
            [
                batch_label,
                ubatch_label,
                max_tokens,
                concurrency,
                throughput,
                total_tokens,
                elapsed,
                errors,
            ]
        )
        results_file.flush()
        completed += 1
        if total_runs:
            elapsed_s = time.time() - sweep_start
            print(
                "progress "
                f"{completed}/{total_runs} "
                f"({completed / total_runs * 100:.1f}%) "
                f"elapsed={elapsed_s:.1f}s "
                f"last=batch={batch_label} "
                f"ubatch={ubatch_label} "
                f"max_tokens={max_tokens} "
                f"concurrency={concurrency}",
                file=sys.stderr,
            )

    best = {
        "throughput": 0.0,
        "tokens": None,
        "concurrency": None,
        "batch": None,
        "ubatch": None,
    }

    def run_cells(proxy, batch_label, ubatch_label, tokens_subset, col_width):
        """Run sweep cells for given max_tokens list; return normally (exceptions propagate)."""
        for max_tokens in tokens_subset:
            row = [str(max_tokens).rjust(15)]
            for concurrency in concurrency_list:
                if total_requests_env:
                    total_requests = int(total_requests_env)
                else:
                    total_requests = max(1, concurrency * requests_multiplier)
                try:
                    result = run_batch(
                        proxy["base_url"],
                        prompt,
                        max_tokens,
                        concurrency,
                        total_requests,
                        temperature,
                    )
                except Exception as exc:
                    print(
                        "error "
                        f"batch={batch_label} ubatch={ubatch_label} "
                        f"max_tokens={max_tokens} concurrency={concurrency}: {exc}",
                        file=sys.stderr,
                    )
                    if not continue_on_error:
                        raise
                    result = {
                        "throughput": 0.0,
                        "total_tokens": 0,
                        "elapsed": 0.0,
                        "errors": total_requests,
                        "last_error": exc,
                    }
                if result["errors"] and result["last_error"]:
                    print(
                        "error "
                        f"batch={batch_label} ubatch={ubatch_label} "
                        f"max_tokens={max_tokens} concurrency={concurrency}: "
                        f"{result['last_error']}",
                        file=sys.stderr,
                    )
                record_row(
                    batch_label,
                    ubatch_label,
                    max_tokens,
                    concurrency,
                    f"{result['throughput']:.1f}",
                    str(result["total_tokens"]),
                    f"{result['elapsed']:.2f}",
                    str(result["errors"]),
                )
                if result["throughput"] > best["throughput"]:
                    best["throughput"] = result["throughput"]
                    best["tokens"] = max_tokens
                    best["concurrency"] = concurrency
                    best["batch"] = batch_label
                    best["ubatch"] = ubatch_label
                row.append(_format_cell(result["throughput"], col_width))
                if cell_pause_s > 0:
                    time.sleep(cell_pause_s)
            print(" ".join(row))

    def record_zeros(batch_label, ubatch_label, tokens_subset):
        for max_tokens in tokens_subset:
            for concurrency in concurrency_list:
                n = int(total_requests_env) if total_requests_env else max(1, concurrency * requests_multiplier)
                record_row(batch_label, ubatch_label, max_tokens, concurrency, "0.0", "0", "0.00", str(n))

    try:
        for batch_size in batch_list:
            for ubatch_size in ubatch_list:
                extra_args = _build_server_args(base_args, batch_size, ubatch_size)
                batch_label = "default" if batch_size is None else str(batch_size)
                ubatch_label = "default" if ubatch_size is None else str(ubatch_size)
                col_width = max(7, max(len(str(c)) for c in concurrency_list))
                header = ["max_tokens \\ conc".rjust(15)] + [str(c).rjust(col_width) for c in concurrency_list]

                # --- Low tokens (≤2048): one server run, ctx = 2048 * parallel ---
                if max_tokens_low:
                    try:
                        os.environ["LLAMA_CTXSIZE_PER_SESSION"] = "2048"
                        os.environ["LLAMA_PARALLEL"] = str(parallel)
                        with start_llama_servers(
                            instance_count,
                            base_port=base_port,
                            ready_timeout_s=ready_timeout_s,
                            startup_delay_s=startup_delay_s,
                            extra_args=extra_args,
                        ) as servers:
                            upstreams = [(s["host"], s["port"]) for s in servers]
                            with start_nginx_round_robin(
                                upstreams,
                                listen_port=nginx_port,
                                listen_host=servers[0]["host"],
                            ) as proxy:
                                if warmup_requests > 0:
                                    for _ in range(warmup_requests):
                                        post_json_with_retry(
                                            f"{proxy['base_url']}/completion",
                                            {"prompt": "warmup", "n_predict": 8, "temperature": 0.0, "stream": False},
                                        )
                                print(f"\nbatch={batch_label} ubatch={ubatch_label} (ctx=2048*parallel, max_tokens<=2048)")
                                print(" ".join(header))
                                print("-" * (len(header) * (col_width + 1)))
                                run_cells(proxy, batch_label, ubatch_label, max_tokens_low, col_width)
                    except Exception as exc:
                        print(f"error batch={batch_label} ubatch={ubatch_label}: {exc}", file=sys.stderr)
                        if not continue_on_error:
                            raise
                        record_zeros(batch_label, ubatch_label, max_tokens_low)

                # --- High tokens (>2048): restart server per max_tokens, ctx = max_tokens * parallel ---
                for max_tokens in max_tokens_high:
                    try:
                        os.environ["LLAMA_CTXSIZE_PER_SESSION"] = str(max_tokens)
                        os.environ["LLAMA_PARALLEL"] = str(parallel)
                        with start_llama_servers(
                            instance_count,
                            base_port=base_port,
                            ready_timeout_s=ready_timeout_s,
                            startup_delay_s=startup_delay_s,
                            extra_args=extra_args,
                        ) as servers:
                            upstreams = [(s["host"], s["port"]) for s in servers]
                            with start_nginx_round_robin(
                                upstreams,
                                listen_port=nginx_port,
                                listen_host=servers[0]["host"],
                            ) as proxy:
                                if warmup_requests > 0:
                                    for _ in range(warmup_requests):
                                        post_json_with_retry(
                                            f"{proxy['base_url']}/completion",
                                            {"prompt": "warmup", "n_predict": 8, "temperature": 0.0, "stream": False},
                                        )
                                print(f"\nbatch={batch_label} ubatch={ubatch_label} (ctx={max_tokens}*parallel)")
                                print(" ".join(header))
                                print("-" * (len(header) * (col_width + 1)))
                                run_cells(proxy, batch_label, ubatch_label, [max_tokens], col_width)
                    except Exception as exc:
                        print(f"error batch={batch_label} ubatch={ubatch_label} max_tokens={max_tokens}: {exc}", file=sys.stderr)
                        if not continue_on_error:
                            raise
                        record_zeros(batch_label, ubatch_label, [max_tokens])
    finally:
        results_file.close()

    print(
        "best "
        f"max_tokens={best['tokens']} "
        f"concurrency={best['concurrency']} "
        f"batch={best['batch']} "
        f"ubatch={best['ubatch']} "
        f"throughput_tps={best['throughput']:.1f}"
    )


if __name__ == "__main__":
    main()
