import os
import sys
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress ResourceWarning for unclosed sockets in threaded urllib use (Python 3.12)
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed .*socket")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tests.llama_server_test_utils import (
    extract_token_count,
    extract_tokens_per_second,
    parse_comma_args,
    post_json,
    start_llama_server,
)


def _parse_int_list(value, default):
    raw = value or default
    parts = [item.strip() for item in raw.replace(",", " ").split()]
    return [int(item) for item in parts if item]


def _parse_optional_int_list(value):
    if value is None:
        return [None]
    parts = [item.strip() for item in value.replace(",", " ").split()]
    result = []
    for item in parts:
        if not item:
            continue
        if item.lower() == "default":
            result.append(None)
        else:
            result.append(int(item))
    return result or [None]


class LlamaServerThreadsSweepTest(unittest.TestCase):
    def test_threads_sweep_throughput(self):
        prompt = os.environ.get(
            "LLAMA_PROMPT",
            "List five ways to make inference servers faster.",
        )
        n_predict = int(os.environ.get("LLAMA_N_PREDICT", "96"))
        total_requests = int(os.environ.get("LLAMA_NUM_REQUESTS", "8"))
        concurrency = int(os.environ.get("LLAMA_CONCURRENCY", "4"))
        threads_list = _parse_int_list(
            os.environ.get("LLAMA_THREADS_LIST"),
            "1,2,4,8,16",
        )
        threads_http_list = _parse_optional_int_list(
            os.environ.get("LLAMA_THREADS_HTTP_LIST")
        )
        threads_http = os.environ.get("LLAMA_THREADS_HTTP")
        if threads_http is not None:
            threads_http_list = _parse_optional_int_list(threads_http)
        base_args = parse_comma_args(os.environ.get("LLAMA_SERVER_ARGS", ""))

        best = {"throughput": 0.0, "threads": None, "threads_http": None}

        for threads in threads_list:
            for threads_http_value in threads_http_list:
                extra_args = base_args + ["--threads", str(threads)]
                if threads_http_value is not None:
                    extra_args += ["--threads-http", str(threads_http_value)]

                with start_llama_server(extra_args=extra_args) as server:
                    start_time = time.time()
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        futures = [
                            executor.submit(
                                post_json,
                                f"{server['base_url']}/completion",
                                {
                                    "prompt": prompt,
                                    "n_predict": n_predict,
                                    "temperature": 0.3,
                                    "stream": False,
                                },
                            )
                            for _ in range(total_requests)
                        ]
                        results = [
                            future.result() for future in as_completed(futures)
                        ]
                    total_time = time.time() - start_time

                total_tokens = sum(
                    extract_token_count(result) for result in results
                )
                per_request_tps = [
                    extract_tokens_per_second(result) for result in results
                ]
                avg_request_tps = (
                    sum(per_request_tps) / len(per_request_tps)
                    if per_request_tps
                    else 0.0
                )
                throughput = total_tokens / total_time if total_time > 0 else 0.0

                self.assertGreater(
                    total_tokens, 0, "Expected tokens from responses."
                )
                self.assertGreater(
                    throughput, 0.0, "Expected throughput > 0."
                )

                if throughput > best["throughput"]:
                    best = {
                        "throughput": throughput,
                        "threads": threads,
                        "threads_http": threads_http_value,
                    }

                threads_http_label = (
                    threads_http_value
                    if threads_http_value is not None
                    else "default"
                )
                print(
                    "threads_sweep "
                    f"threads={threads} "
                    f"threads_http={threads_http_label} "
                    f"requests={total_requests} concurrency={concurrency} "
                    f"total_tokens={total_tokens} "
                    f"avg_request_tps={avg_request_tps:.2f} "
                    f"throughput_tps={throughput:.2f}"
                )

        print(
            "threads_sweep_best "
            f"threads={best['threads']} "
            f"threads_http={best['threads_http']} "
            f"throughput_tps={best['throughput']:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
