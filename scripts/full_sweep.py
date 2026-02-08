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


def parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]

def parse_optional_int_list(value, default):
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

def build_server_args(base_args, parallel, batch_size, ubatch_size):
    args = parse_comma_args(base_args)

    cleaned = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--parallel", "--batch-size", "--ubatch", "-b"}:
            skip_next = True
            continue
        if arg.startswith("--parallel="):
            continue
        if arg.startswith("--batch-size="):
            continue
        if arg.startswith("--ubatch="):
            continue
        cleaned.append(arg)

    cleaned += ["--parallel", str(parallel)]
    if batch_size is not None:
        cleaned += ["--batch-size", str(batch_size)]
    if ubatch_size is not None:
        cleaned += ["--ubatch", str(ubatch_size)]
    return cleaned


def post_json_with_retry(url, payload, timeout, max_attempts, base_sleep_s):
    for attempt in range(max_attempts):
        try:
            return post_json(url, payload, timeout=timeout)
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


def run_batch(
    base_url,
    prompt,
    n_predict,
    concurrency,
    total_requests,
    temperature,
    request_timeout,
    retry_attempts,
    retry_sleep_s,
):
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
                request_timeout,
                retry_attempts,
                retry_sleep_s,
            )
            for _ in range(total_requests)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                errors += 1
                last_error = exc

    elapsed = time.time() - start_time
    total_tokens = sum(extract_token_count(result) for result in results)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "throughput": throughput,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "errors": errors,
        "last_error": last_error,
    }


def main():
    prompt = os.environ.get(
        "LLAMA_PROMPT",
        "Share three optimization tips for model serving.",
    )
    temperature = float(os.environ.get("LLAMA_TEMPERATURE", "0.3"))
    n_predict = int(os.environ.get("LLAMA_N_PREDICT", "128"))

    instances_list = parse_int_list(
        os.environ.get("LLAMA_INSTANCES_LIST"),
        "2,4,8,16",
    )
    parallel_list = parse_int_list(
        os.environ.get("LLAMA_PARALLEL_LIST"),
        "1,2,4,8,16,32,64",
    )
    batch_list = parse_optional_int_list(
        os.environ.get("LLAMA_BATCH_LIST"),
        "default",
    )
    ubatch_list = parse_optional_int_list(
        os.environ.get("LLAMA_UBATCH_LIST"),
        "default",
    )
    concurrency_list = parse_int_list(
        os.environ.get("LLAMA_CONCURRENCY_LIST"),
        "1,2,4,8,16,32,64,128,256,512,1024",
    )

    base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
    nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
    base_args = os.environ.get("LLAMA_SERVER_ARGS", "")

    ready_timeout_s = int(os.environ.get("LLAMA_READY_TIMEOUT", "180"))
    startup_delay_s = float(os.environ.get("LLAMA_STARTUP_DELAY_S", "0.0"))
    warmup_requests = int(os.environ.get("LLAMA_WARMUP_REQUESTS", "2"))
    request_timeout = float(os.environ.get("LLAMA_REQUEST_TIMEOUT", "120"))
    retry_attempts = int(os.environ.get("LLAMA_RETRY_ATTEMPTS", "8"))
    retry_sleep_s = float(os.environ.get("LLAMA_RETRY_SLEEP_S", "0.5"))
    cell_pause_s = float(os.environ.get("LLAMA_CELL_PAUSE_S", "0.0"))
    continue_on_error = os.environ.get("LLAMA_CONTINUE_ON_ERROR", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    requests_multiplier = int(os.environ.get("LLAMA_REQUESTS_MULTIPLIER", "1"))
    total_requests_env = os.environ.get("LLAMA_NUM_REQUESTS")

    if requests_multiplier < 1:
        requests_multiplier = 1

    results_path = init_results_file("full_sweep", "full_sweep")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_file = results_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(results_file)
    writer.writerow(
        [
            "instances",
            "parallel",
            "batch",
            "ubatch",
            "concurrency",
            "throughput_tps",
            "total_tokens",
            "elapsed_s",
            "errors",
        ]
    )
    results_file.flush()

    print(
        "instances,parallel,batch,ubatch,concurrency,throughput_tps,"
        "total_tokens,elapsed_s,errors"
    )
    print(f"results_file={results_path}")

    total_runs = (
        len(instances_list)
        * len(parallel_list)
        * len(batch_list)
        * len(ubatch_list)
        * len(concurrency_list)
    )
    completed = 0
    sweep_start = time.time()

    def record_row(
        instances,
        parallel,
        batch_label,
        ubatch_label,
        concurrency,
        throughput,
        total_tokens,
        elapsed,
        errors,
    ):
        nonlocal completed
        writer.writerow(
            [
                instances,
                parallel,
                batch_label,
                ubatch_label,
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
                f"last=instances={instances} "
                f"parallel={parallel} "
                f"batch={batch_label} "
                f"ubatch={ubatch_label} "
                f"concurrency={concurrency}",
                file=sys.stderr,
            )

    best = {
        "throughput": 0.0,
        "instances": None,
        "parallel": None,
        "batch": None,
        "ubatch": None,
        "concurrency": None,
    }

    try:
        for instances in instances_list:
            for parallel in parallel_list:
                for batch_size in batch_list:
                    for ubatch_size in ubatch_list:
                        server_args = build_server_args(
                            base_args, parallel, batch_size, ubatch_size
                        )
                        batch_label = (
                            "default" if batch_size is None else str(batch_size)
                        )
                        ubatch_label = (
                            "default" if ubatch_size is None else str(ubatch_size)
                        )
                        try:
                            os.environ["LLAMA_PARALLEL"] = str(parallel)
                            with start_llama_servers(
                                instances,
                                base_port=base_port,
                                extra_args=server_args,
                                ready_timeout_s=ready_timeout_s,
                                startup_delay_s=startup_delay_s,
                            ) as servers:
                                upstreams = [
                                    (server["host"], server["port"])
                                    for server in servers
                                ]
                                with start_nginx_round_robin(
                                    upstreams,
                                    listen_port=nginx_port,
                                    listen_host=servers[0]["host"],
                                ) as proxy:
                                    if warmup_requests > 0:
                                        for _ in range(warmup_requests):
                                            post_json_with_retry(
                                                f"{proxy['base_url']}/completion",
                                                {
                                                    "prompt": "warmup",
                                                    "n_predict": 8,
                                                    "temperature": 0.0,
                                                    "stream": False,
                                                },
                                                request_timeout,
                                                retry_attempts,
                                                retry_sleep_s,
                                            )

                                    for concurrency in concurrency_list:
                                        if total_requests_env:
                                            total_requests = int(total_requests_env)
                                        else:
                                            total_requests = max(
                                                1, concurrency * requests_multiplier
                                            )
                                        try:
                                            result = run_batch(
                                                proxy["base_url"],
                                                prompt,
                                                n_predict,
                                                concurrency,
                                                total_requests,
                                                temperature,
                                                request_timeout,
                                                retry_attempts,
                                                retry_sleep_s,
                                            )
                                        except Exception as exc:
                                            print(
                                                "error "
                                                f"instances={instances} "
                                                f"parallel={parallel} "
                                                f"batch={batch_label} "
                                                f"ubatch={ubatch_label} "
                                                f"concurrency={concurrency}: {exc}",
                                                file=sys.stderr,
                                            )
                                            if not continue_on_error:
                                                raise
                                            print(
                                                "failing_row "
                                                f"{instances},{parallel},{batch_label},"
                                                f"{ubatch_label},{concurrency},"
                                                "0.0,0,0.00,"
                                                f"{total_requests}"
                                            )
                                            record_row(
                                                instances,
                                                parallel,
                                                batch_label,
                                                ubatch_label,
                                                concurrency,
                                                "0.0",
                                                "0",
                                                "0.00",
                                                str(total_requests),
                                            )
                                            if cell_pause_s > 0:
                                                time.sleep(cell_pause_s)
                                            continue

                                        print(
                                            f"{instances},{parallel},{batch_label},"
                                            f"{ubatch_label},{concurrency},"
                                            f"{result['throughput']:.1f},"
                                            f"{result['total_tokens']},"
                                            f"{result['elapsed']:.2f},"
                                            f"{result['errors']}"
                                        )
                                        record_row(
                                            instances,
                                            parallel,
                                            batch_label,
                                            ubatch_label,
                                            concurrency,
                                            f"{result['throughput']:.1f}",
                                            str(result["total_tokens"]),
                                            f"{result['elapsed']:.2f}",
                                            str(result["errors"]),
                                        )
                                        if result["throughput"] > best["throughput"]:
                                            best = {
                                                "throughput": result["throughput"],
                                                "instances": instances,
                                                "parallel": parallel,
                                                "batch": batch_label,
                                                "ubatch": ubatch_label,
                                                "concurrency": concurrency,
                                            }
                                        if cell_pause_s > 0:
                                            time.sleep(cell_pause_s)
                        except Exception as exc:
                            print(
                                "error "
                                f"instances={instances} "
                                f"parallel={parallel} "
                                f"batch={batch_label} "
                                f"ubatch={ubatch_label}: {exc}",
                                file=sys.stderr,
                            )
                            if not continue_on_error:
                                raise
                            for concurrency in concurrency_list:
                                if total_requests_env:
                                    total_requests = int(total_requests_env)
                                else:
                                    total_requests = max(
                                        1, concurrency * requests_multiplier
                                    )
                                record_row(
                                    instances,
                                    parallel,
                                    batch_label,
                                    ubatch_label,
                                    concurrency,
                                    "0.0",
                                    "0",
                                    "0.00",
                                    str(total_requests),
                                )
                            continue
    finally:
        results_file.close()

    print(
        "best "
        f"instances={best['instances']} "
        f"parallel={best['parallel']} "
        f"batch={best['batch']} "
        f"ubatch={best['ubatch']} "
        f"concurrency={best['concurrency']} "
        f"throughput_tps={best['throughput']:.1f}"
    )


if __name__ == "__main__":
    main()
