import contextlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def parse_comma_args(raw_args):
    if not raw_args:
        return []
    return [arg.strip() for arg in raw_args.split(",") if arg.strip()]


def _find_llama_cpp_dir():
    search_roots = [REPO_ROOT, *REPO_ROOT.parents]
    for base in search_roots:
        if base.name == "llama.cpp":
            if (
                (base / "build" / "bin" / "llama-server").is_file()
                or (base / "build" / "bin" / "server").is_file()
                or (base / "llama-server").is_file()
                or (base / "server").is_file()
            ):
                return str(base)

        candidate = base / "llama.cpp"
        if candidate.is_dir():
            if (
                (candidate / "build" / "bin" / "llama-server").is_file()
                or (candidate / "build" / "bin" / "server").is_file()
                or (candidate / "llama-server").is_file()
                or (candidate / "server").is_file()
            ):
                return str(candidate)

    return ""


def resolve_llama_cpp_dir():
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        return env_dir

    detected = _find_llama_cpp_dir()
    if detected:
        return detected

    return "llama.cpp"


def resolve_model_path():
    return os.environ.get("LLAMA_MODEL_PATH", "")


def resolve_llama_server_bin():
    env_path = os.environ.get("LLAMA_SERVER_BIN")
    if env_path:
        return env_path

    llama_cpp_dir = resolve_llama_cpp_dir()
    candidates = [
        os.path.join(llama_cpp_dir, "llama-server"),
        os.path.join(llama_cpp_dir, "server"),
        os.path.join(llama_cpp_dir, "build", "bin", "llama-server"),
        os.path.join(llama_cpp_dir, "build", "bin", "server"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return candidates[0]


def _pick_port(allow_env_port=True):
    env_port = os.environ.get("LLAMA_SERVER_PORT") if allow_env_port else None
    if env_port:
        return int(env_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return sock.getsockname()[1]


def _wait_for_server(host, port, timeout_s=None):
    if timeout_s is None:
        timeout_s = int(os.environ.get("LLAMA_SERVER_BIND_TIMEOUT", "180"))
    deadline = time.time() + timeout_s
    last_error = None
    health_url = f"http://{host}:{port}/health"
    models_url = f"http://{host}:{port}/v1/models"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in {200, 404}:
                return
            last_error = exc
        except Exception as exc:
            last_error = exc

        try:
            with urllib.request.urlopen(models_url, timeout=2) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code in {200, 404}:
                return
            last_error = exc
        except Exception as exc:
            last_error = exc

        time.sleep(0.5)

    raise RuntimeError(
        f"Server did not become ready at {host}:{port} within {timeout_s}s: {last_error}. "
        "Check that the port is free (e.g. stop any round-robin servers) and that the model "
        "loads in time (try increasing LLAMA_SERVER_BIND_TIMEOUT or LLAMA_READY_TIMEOUT)."
    )


def _wait_for_completion_ready(host, port, timeout_s=120):
    deadline = time.time() + timeout_s
    last_error = None
    url = f"http://{host}:{port}/completion"
    payload = {
        "prompt": "ping",
        "n_predict": 1,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")

    while time.time() < deadline:
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status == 200:
                    resp.read()
                    resp.close()
                    return
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                time.sleep(0.5)
                continue
            with exc:
                data = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP error {exc.code}: {data}") from exc
        except Exception as exc:
            time.sleep(0.5)
            last_error = exc

    raise RuntimeError(f"Model did not become ready: {last_error}")


@contextlib.contextmanager
def start_llama_server(port=None, host=None, extra_args=None, ready_timeout_s=None):
    server_bin = resolve_llama_server_bin()
    model_path = resolve_model_path()
    if not os.path.isfile(server_bin):
        raise FileNotFoundError(
            f"llama-server binary not found at {server_bin}. "
            "Set LLAMA_SERVER_BIN or build llama.cpp."
        )
    if not model_path:
        raise FileNotFoundError(
            "Model path not set. Set LLAMA_MODEL_PATH or use the launcher."
        )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Set LLAMA_MODEL_PATH."
        )

    if host is None:
        host = os.environ.get("LLAMA_SERVER_HOST", DEFAULT_HOST)
    if port is None:
        port = _pick_port()
    else:
        port = int(port)
    if extra_args is None:
        extra_args = parse_comma_args(os.environ.get("LLAMA_SERVER_ARGS", ""))

    # Always set --ctx-size so we don't allocate too much memory.
    # ctx_size = ctxsize_per_session * parallel; use n_predict when CTXSIZE_PER_SESSION not set.
    ctxsize_per_session = int(
        os.environ.get("LLAMA_CTXSIZE_PER_SESSION")
        or os.environ.get("LLAMA_N_PREDICT", "2048")
    )
    parallel = int(os.environ.get("LLAMA_PARALLEL", "1"))
    ctx_size = ctxsize_per_session * parallel

    cmd = [
        server_bin,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_path,
        "--ctx-size",
        str(ctx_size),
        "--parallel",
        str(parallel),
    ] + extra_args

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        bind_timeout = int(os.environ.get("LLAMA_SERVER_BIND_TIMEOUT", "180"))
        _wait_for_server(host, port, timeout_s=bind_timeout)
        completion_timeout = ready_timeout_s
        if completion_timeout is None:
            completion_timeout = int(os.environ.get("LLAMA_READY_TIMEOUT", "120"))
        _wait_for_completion_ready(host, port, timeout_s=completion_timeout)
        yield {
            "host": host,
            "port": port,
            "base_url": f"http://{host}:{port}",
            "process": process,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextlib.contextmanager
def start_llama_servers(
    count,
    base_port,
    host=None,
    extra_args=None,
    ready_timeout_s=None,
    startup_delay_s=None,
):
    if count < 1:
        raise ValueError("count must be >= 1")
    if base_port is None:
        base_port = _pick_port(allow_env_port=False)

    servers = []
    with contextlib.ExitStack() as stack:
        for index in range(count):
            port = base_port + index
            servers.append(
                stack.enter_context(
                    start_llama_server(
                        port=port,
                        host=host,
                        extra_args=extra_args,
                        ready_timeout_s=ready_timeout_s,
                    )
                )
            )
            if startup_delay_s:
                time.sleep(startup_delay_s)
        yield servers


def resolve_nginx_bin():
    env_path = os.environ.get("NGINX_BIN")
    if env_path:
        return env_path
    return "nginx"


def _wait_for_port(host, port, timeout_s=20):
    deadline = time.time() + timeout_s
    last_error = None

    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            try:
                sock.connect((host, port))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)

    raise RuntimeError(f"Port {port} did not become ready: {last_error}")


@contextlib.contextmanager
def start_nginx_round_robin(upstreams, listen_port, listen_host=None):
    nginx_bin = resolve_nginx_bin()
    if not (os.path.isfile(nginx_bin) or shutil.which(nginx_bin)):
        raise FileNotFoundError(
            "nginx binary not found. Install nginx or set NGINX_BIN."
        )

    if listen_host is None:
        listen_host = DEFAULT_HOST

    temp_dir = tempfile.TemporaryDirectory()
    conf_path = os.path.join(temp_dir.name, "nginx.conf")
    upstream_lines = "\n".join(
        [f"        server {host}:{port};" for host, port in upstreams]
    )
    conf = (
        "worker_processes 1;\n"
        f"pid {temp_dir.name}/nginx.pid;\n"
        f"error_log {temp_dir.name}/error.log;\n"
        "events { worker_connections 1024; }\n"
        "http {\n"
        f"    access_log {temp_dir.name}/access.log;\n"
        "    upstream llama_backend {\n"
        f"{upstream_lines}\n"
        "    }\n"
        "    server {\n"
        f"        listen {listen_host}:{listen_port};\n"
        "        location / {\n"
        "            proxy_pass http://llama_backend;\n"
        "            proxy_http_version 1.1;\n"
        "            proxy_set_header Connection \"\";\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    with open(conf_path, "w", encoding="utf-8") as handle:
        handle.write(conf)

    process = subprocess.Popen(
        [nginx_bin, "-c", conf_path, "-p", temp_dir.name, "-g", "daemon off;"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_port(listen_host, listen_port)
        yield {
            "host": listen_host,
            "port": listen_port,
            "base_url": f"http://{listen_host}:{listen_port}",
            "process": process,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        temp_dir.cleanup()


def post_json(url, payload, timeout=120):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
            resp.close()
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        with exc:
            data = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error {exc.code}: {data}") from exc


def extract_token_count(response):
    timings = response.get("timings") or {}
    for key in ("predicted_n", "tokens_predicted", "completion_tokens"):
        if key in timings:
            return int(timings[key])
        if key in response:
            return int(response[key])

    usage = response.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])

    return 0


def extract_tokens_per_second(response):
    timings = response.get("timings") or {}
    for key in ("predicted_per_second", "tokens_per_second"):
        if key in timings:
            return float(timings[key])

    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    if predicted_n and predicted_ms:
        return float(predicted_n) / (float(predicted_ms) / 1000.0)

    return 0.0
