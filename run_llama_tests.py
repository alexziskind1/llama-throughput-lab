#!/usr/bin/env python3
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()

TESTS = {
    "1": ("Tests: Single request", ["-m", "unittest", "tests/test_llama_server_single.py"]),
    "2": ("Tests: Concurrent requests", ["-m", "unittest", "tests/test_llama_server_concurrent.py"]),
    "3": ("Tests: Round-robin (nginx + multiple servers)", ["-m", "unittest", "tests/test_llama_server_round_robin.py"]),
    "4": ("Sweeps: Threads (--threads/--threads-http)", ["tests/test_llama_server_threads_sweep.py"]),
    "5": ("Sweeps: Round-robin (max_tokens x concurrency)", ["scripts/round_robin_sweep.py"]),
    "6": ("Sweeps: Full (instances x parallel x concurrency)", ["scripts/full_sweep.py"]),
}


def check_dependencies():
    if not shutil.which("dialog"):
        print("Error: 'dialog' is required. Install it (e.g., brew install dialog).")
        sys.exit(1)


def warn_if_missing_nginx():
    if shutil.which("nginx"):
        return
    show_msg(
        "Missing nginx",
        "nginx was not found in PATH.\n"
        "Round-robin tests/sweeps will fail until nginx is installed.",
    )


def run_dialog(args):
    with tempfile.NamedTemporaryFile(mode="w+") as tf:
        cmd = ["dialog"] + args
        try:
            res = subprocess.run(cmd, stderr=tf, check=False)
            tf.seek(0)
            return tf.read().strip(), res.returncode
        except Exception:
            return None, 1


def show_msg(title, msg):
    run_dialog(["--title", title, "--msgbox", msg, "10", "60"])


def get_directory_contents(path):
    try:
        if not os.path.isdir(path):
            return [], []

        entries = os.listdir(path)
        dirs = []
        files = []

        for entry in entries:
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            elif entry.endswith(".gguf"):
                files.append(entry)

        dirs.sort()
        files.sort()
        return dirs, files
    except PermissionError:
        return [], []


def custom_file_picker(start_path):
    current_path = os.path.abspath(start_path)
    if not os.path.isdir(current_path):
        current_path = os.getcwd()

    while True:
        dirs, files = get_directory_contents(current_path)
        menu_items = []

        if current_path != "/":
            menu_items.extend(["..", "Parent Directory"])

        for directory in dirs:
            menu_items.extend([directory + "/", "<DIR>"])

        for file_name in files:
            menu_items.extend([file_name, "<GGUF>"])

        if not menu_items:
            menu_items.extend([".", "Empty Directory"])

        pretty_path = current_path
        if len(pretty_path) > 50:
            pretty_path = "..." + pretty_path[-47:]

        selection, code = run_dialog(
            [
                "--title",
                "Select GGUF File",
                "--backtitle",
                f"Current: {pretty_path}",
                "--menu",
                "Navigate directories and select a .gguf file:",
                "20",
                "70",
                "12",
                *menu_items,
            ]
        )

        if code != 0:
            return None

        clean = selection.strip()
        if clean == "..":
            current_path = os.path.dirname(current_path)
        elif clean.endswith("/"):
            current_path = os.path.join(current_path, clean[:-1])
        elif clean == ".":
            pass
        else:
            return os.path.join(current_path, clean)


def _parse_model_dirs(raw):
    if not raw:
        return []
    parts = []
    for chunk in raw.split(os.pathsep):
        for item in chunk.split(","):
            item = item.strip()
            if item:
                parts.append(os.path.expanduser(item))
    return parts


def _find_gguf_in_dir(base_dir, max_depth):
    if not base_dir or not os.path.isdir(base_dir):
        return None, None

    best_path = None
    best_mtime = None
    base_dir = os.path.abspath(base_dir)

    for root, dirs, files in os.walk(base_dir):
        rel_path = os.path.relpath(root, base_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        if depth >= max_depth:
            dirs[:] = []

        for file_name in files:
            if not file_name.endswith(".gguf"):
                continue
            path = os.path.join(root, file_name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if best_mtime is None or mtime > best_mtime:
                best_mtime = mtime
                best_path = path

    return best_path, best_mtime


def auto_detect_model():
    env_model = os.environ.get("LLAMA_MODEL_PATH")
    if env_model and os.path.isfile(env_model):
        return env_model

    model_dirs = _parse_model_dirs(os.environ.get("LLAMA_MODEL_DIRS"))
    if not model_dirs:
        home = str(Path.home())
        model_dirs = [
            os.path.join(SCRIPT_DIR, "models"),
            os.path.join(SCRIPT_DIR, "llama.cpp", "models"),
            "/models",
            os.path.join(home, "models"),
            os.path.join(home, "Models"),
            os.path.join(home, "Downloads"),
            os.path.join(home, ".cache", "lm-studio", "models"),
        ]

    max_depth = int(os.environ.get("LLAMA_MODEL_SEARCH_DEPTH", "4"))
    best_path = None
    best_mtime = None

    for base_dir in model_dirs:
        path, mtime = _find_gguf_in_dir(base_dir, max_depth)
        if path and (best_mtime is None or mtime > best_mtime):
            best_path = path
            best_mtime = mtime

    return best_path or ""


def _find_llama_cpp_dir():
    search_roots = [SCRIPT_DIR, *SCRIPT_DIR.parents]
    for base in search_roots:
        if base.name == "llama.cpp":
            return str(base)
        candidate = base / "llama.cpp"
        if candidate.is_dir():
            return str(candidate)
    return ""


def auto_detect_server_bin():
    env_bin = os.environ.get("LLAMA_SERVER_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin

    cpp_dir = os.environ.get("LLAMA_CPP_DIR") or _find_llama_cpp_dir()
    candidates = []
    if cpp_dir:
        candidates.extend(
            [
                os.path.join(cpp_dir, "build", "bin", "llama-server"),
                os.path.join(cpp_dir, "build", "bin", "server"),
                os.path.join(cpp_dir, "llama-server"),
                os.path.join(cpp_dir, "server"),
            ]
        )

    candidates.append(
        os.path.join(SCRIPT_DIR, "llama.cpp", "build", "bin", "llama-server")
    )
    candidates.append(shutil.which("llama-server") or "")

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return ""


def find_python():
    venv_python = SCRIPT_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python3"


def parse_env_overrides(raw):
    overrides = {}
    if not raw:
        return overrides
    for token in shlex.split(raw):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key:
            overrides[key] = value
    return overrides


class AppState:
    def __init__(self):
        self.model_path = auto_detect_model()
        self.server_bin = auto_detect_server_bin()
        self.env_overrides = ""
        self.test_key = "1"
        self.n_predict = int(os.environ.get("LLAMA_N_PREDICT", "128"))
        self.max_tokens_list = os.environ.get("LLAMA_MAX_TOKENS_LIST", "128,256,512,1024")
        self.concurrency_list = os.environ.get(
            "LLAMA_CONCURRENCY_LIST", "1,2,4,8,16,32,64,128,256,512,1024"
        )
        self.rr_instances = int(os.environ.get("LLAMA_SERVER_INSTANCES", "2"))
        self.rr_parallel = int(os.environ.get("LLAMA_PARALLEL", "16"))
        self.rr_base_port = int(os.environ.get("LLAMA_SERVER_BASE_PORT", "9000"))
        self.rr_nginx_port = int(os.environ.get("LLAMA_NGINX_PORT", "8088"))
        self.rr_host = os.environ.get("LLAMA_SERVER_HOST", "127.0.0.1")
        self.advanced_args = os.environ.get("LLAMA_SERVER_ARGS", "")

    @property
    def test_label(self):
        return TESTS[self.test_key][0]


def select_test(state):
    items = []
    for key in sorted(TESTS.keys(), key=int):
        items.extend([key, TESTS[key][0]])
    selection, code = run_dialog(
        [
            "--title",
            "Select Test",
            "--menu",
            "Choose a test or sweep to run:",
            "18",
            "70",
            "8",
            *items,
        ]
    )
    if code == 0 and selection:
        state.test_key = selection


def select_model(state):
    start_path = state.model_path or os.getcwd()
    if os.path.isfile(start_path):
        start_path = os.path.dirname(start_path)
    selection = custom_file_picker(start_path)
    if selection:
        state.model_path = selection


def edit_env_overrides(state):
    selection, code = run_dialog(
        [
            "--title",
            "Environment Overrides",
            "--inputbox",
            "Enter KEY=VALUE pairs separated by spaces.\n"
            "Example: LLAMA_CONCURRENCY=64 LLAMA_NUM_REQUESTS=64",
            "12",
            "70",
            state.env_overrides or "",
        ]
    )
    if code == 0:
        state.env_overrides = selection.strip()


def edit_n_predict(state):
    current = str(state.n_predict)
    selection, code = run_dialog(
        [
            "--title",
            "Tokens (single test)",
            "--inputbox",
            "Tokens to generate per request (single and non-sweep tests):",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip().isdigit():
        state.n_predict = int(selection.strip())


def edit_max_tokens_list(state):
    current = state.max_tokens_list or "128,256,512,1024"
    selection, code = run_dialog(
        [
            "--title",
            "Tokens (sweep tests)",
            "--inputbox",
            "Comma-separated token counts for sweep (e.g. 128,256,512,1024):",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip():
        state.max_tokens_list = selection.strip()


def edit_concurrency_list(state):
    current = state.concurrency_list or "1,2,4,8,16,32,64,128,256,512,1024"
    selection, code = run_dialog(
        [
            "--title",
            "List of concurrent tests (sweeps)",
            "--inputbox",
            "Comma-separated concurrencies for sweep (e.g. 1,4,8,16,32,64):",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip():
        state.concurrency_list = selection.strip()


def tokens_menu(state):
    while True:
        menu = [
            "--clear",
            "--backtitle",
            "Llama.cpp Local Test Launcher",
            "--title",
            "Tokens and sweep",
            "--menu",
            "Single/sweep tokens, concurrency list.",
            "18",
            "70",
            "4",
            "1",
            f"Single test: {state.n_predict}",
            "2",
            f"Sweep tokens: {state.max_tokens_list}",
            "3",
            f"List of concurrent tests: {state.concurrency_list}",
            "4",
            "Back",
        ]
        choice, code = run_dialog(menu)
        if code != 0:
            break
        if choice == "1":
            edit_n_predict(state)
        elif choice == "2":
            edit_max_tokens_list(state)
        elif choice == "3":
            edit_concurrency_list(state)
        elif choice == "4":
            break


def edit_advanced_args(state):
    selection, code = run_dialog(
        [
            "--title",
            "Advanced Server Arguments",
            "--inputbox",
            "Enter llama-server arguments (e.g., --ctx-size 4096 -fa 1 --mmproj ./model.bin)\n"
            "These will be passed directly to llama-server.\n"
            "Note: --ctx-size and --parallel set here will override computed values.",
            "12",
            "70",
            state.advanced_args or "",
        ]
    )
    if code == 0:
        state.advanced_args = selection.strip()


def edit_server_bin(state):
    current = state.server_bin or ""
    selection, code = run_dialog(
        [
            "--title",
            "llama-server Binary",
            "--inputbox",
            "Enter the path to llama-server.\nLeave empty to rely on auto-detect or PATH.",
            "12",
            "70",
            current,
        ]
    )
    if code == 0:
        state.server_bin = selection.strip()


def run_selected(state):
    python_bin = find_python()
    label, args = TESTS[state.test_key]
    cmd = [python_bin] + args

    overrides = parse_env_overrides(state.env_overrides)
    if state.model_path and "LLAMA_MODEL_PATH" not in overrides:
        overrides["LLAMA_MODEL_PATH"] = state.model_path
    if state.server_bin and "LLAMA_SERVER_BIN" not in overrides:
        overrides["LLAMA_SERVER_BIN"] = state.server_bin
    if state.advanced_args and "LLAMA_SERVER_ARGS" not in overrides:
        overrides["LLAMA_SERVER_ARGS"] = state.advanced_args
    if "LLAMA_N_PREDICT" not in overrides:
        overrides["LLAMA_N_PREDICT"] = str(state.n_predict)
    if state.test_key == "5" and "LLAMA_MAX_TOKENS_LIST" not in overrides:
        overrides["LLAMA_MAX_TOKENS_LIST"] = state.max_tokens_list
    if state.test_key in ("5", "6") and "LLAMA_CONCURRENCY_LIST" not in overrides:
        overrides["LLAMA_CONCURRENCY_LIST"] = state.concurrency_list
    # Only inject LLAMA_PARALLEL for round-robin test/sweep (3, 5). Single/concurrent (1, 2)
    # and threads sweep (4) use test default 1 to avoid changing behavior and memory.
    if state.test_key in ("3", "5") and "LLAMA_PARALLEL" not in overrides:
        overrides["LLAMA_PARALLEL"] = str(state.rr_parallel)

    env = os.environ.copy()
    env.update(overrides)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SCRIPT_DIR), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    subprocess.run(["clear"])
    print("=== Running Llama.cpp Test ===")
    print(f"Test:  {label}")
    if state.model_path:
        print(f"Model: {state.model_path}")
    print("Parameters:")
    print(f"  Client token size (n_predict): {overrides.get('LLAMA_N_PREDICT', str(state.n_predict))}")
    ctx_effective = overrides.get("LLAMA_CTXSIZE_PER_SESSION") or str(state.n_predict)
    ctx_note = " (from n_predict)" if "LLAMA_CTXSIZE_PER_SESSION" not in overrides else ""
    print(f"  Context per session (--ctx-size): {ctx_effective}{ctx_note}")
    print(f"  Sweep tokens:                  {overrides.get('LLAMA_MAX_TOKENS_LIST', state.max_tokens_list)}")
    print(f"  List of concurrent tests:     {overrides.get('LLAMA_CONCURRENCY_LIST', state.concurrency_list)}")
    parallel_display = overrides.get("LLAMA_PARALLEL")
    if parallel_display is None and state.test_key in ("1", "2", "4"):
        parallel_display = "1 (test default)"
    elif parallel_display is None:
        parallel_display = str(state.rr_parallel)
    print(f"  Parallel (server --parallel): {parallel_display}")
    if state.env_overrides:
        print(f"Env:   {state.env_overrides}")
    if state.advanced_args:
        print(f"Advanced Args: {state.advanced_args}")
    print("--------------------------------")
    print(f"CMD: {' '.join(cmd)}")
    print("")

    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), env=env)
    proc.wait()

    input("\nRun complete. Press Enter to return to menu...")


def run_round_robin(state, action):
    script_path = SCRIPT_DIR / "start_llama_rr.sh"
    if not script_path.exists():
        show_msg("Error", "start_llama_rr.sh not found.")
        return

    overrides = parse_env_overrides(state.env_overrides)
    if state.model_path and "LLAMA_MODEL_PATH" not in overrides:
        overrides["LLAMA_MODEL_PATH"] = state.model_path
    if state.server_bin and "LLAMA_SERVER_BIN" not in overrides:
        overrides["LLAMA_SERVER_BIN"] = state.server_bin
    if state.advanced_args and "LLAMA_SERVER_ARGS" not in overrides:
        overrides["LLAMA_SERVER_ARGS"] = state.advanced_args
    overrides.setdefault("LLAMA_SERVER_INSTANCES", str(state.rr_instances))
    overrides.setdefault("LLAMA_PARALLEL", str(state.rr_parallel))
    overrides.setdefault("LLAMA_N_PREDICT", str(state.n_predict))
    overrides.setdefault("LLAMA_SERVER_BASE_PORT", str(state.rr_base_port))
    overrides.setdefault("LLAMA_NGINX_PORT", str(state.rr_nginx_port))
    overrides.setdefault("LLAMA_SERVER_HOST", state.rr_host)

    env = os.environ.copy()
    env.update(overrides)

    subprocess.run(["clear"])
    print("=== Round-robin Servers ===")
    print(f"Action: {action}")
    if state.model_path:
        print(f"Model:  {state.model_path}")
    if state.env_overrides:
        print(f"Env:    {state.env_overrides}")
    print("--------------------------------")
    cmd = [str(script_path), action]
    print(f"CMD: {' '.join(cmd)}")
    print("")

    subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env)
    input("\nDone. Press Enter to return to menu...")


def edit_rr_instances(state):
    current = str(state.rr_instances)
    selection, code = run_dialog(
        [
            "--title",
            "Round-robin Instances",
            "--inputbox",
            "Number of llama-server instances to start:",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip().isdigit():
        state.rr_instances = int(selection.strip())


def edit_rr_parallel(state):
    current = str(state.rr_parallel)
    selection, code = run_dialog(
        [
            "--title",
            "Round-robin Parallel",
            "--inputbox",
            "Value for --parallel:",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip().isdigit():
        state.rr_parallel = int(selection.strip())


def edit_rr_base_port(state):
    current = str(state.rr_base_port)
    selection, code = run_dialog(
        [
            "--title",
            "Round-robin Base Port",
            "--inputbox",
            "Base port for the first llama-server instance:",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip().isdigit():
        state.rr_base_port = int(selection.strip())


def edit_rr_nginx_port(state):
    current = str(state.rr_nginx_port)
    selection, code = run_dialog(
        [
            "--title",
            "Round-robin Nginx Port",
            "--inputbox",
            "Port for nginx to listen on:",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip().isdigit():
        state.rr_nginx_port = int(selection.strip())


def edit_rr_host(state):
    current = state.rr_host
    selection, code = run_dialog(
        [
            "--title",
            "Round-robin Host",
            "--inputbox",
            "Host address for llama-server and nginx:",
            "10",
            "60",
            current,
        ]
    )
    if code == 0 and selection.strip():
        state.rr_host = selection.strip()


def round_robin_menu(state):
    while True:
        menu = [
            "--clear",
            "--backtitle",
            "Llama.cpp Local Test Launcher",
            "--title",
            "Configure and Run Round Robin",
            "--menu",
            "Set parameters and start/stop servers:",
            "22",
            "70",
            "10",
            "1",
            f"Instances: {state.rr_instances}",
            "2",
            f"Parallel: {state.rr_parallel}",
            "3",
            f"Base Port: {state.rr_base_port}",
            "4",
            f"Nginx Port: {state.rr_nginx_port}",
            "5",
            f"Host: {state.rr_host}",
            "6",
            f"Advanced Args: {state.advanced_args or '(None)'}",
            "7",
            "Start round-robin servers",
            "8",
            "Stop round-robin servers",
            "9",
            "Back",
        ]

        choice, code = run_dialog(menu)
        if code != 0:
            break
        if choice == "1":
            edit_rr_instances(state)
        elif choice == "2":
            edit_rr_parallel(state)
        elif choice == "3":
            edit_rr_base_port(state)
        elif choice == "4":
            edit_rr_nginx_port(state)
        elif choice == "5":
            edit_rr_host(state)
        elif choice == "6":
            edit_advanced_args(state)
        elif choice == "7":
            run_round_robin(state, "start")
        elif choice == "8":
            run_round_robin(state, "stop")
        elif choice == "9":
            break


def main_menu():
    state = AppState()

    while True:
        model_display = Path(state.model_path).name if state.model_path else "(None)"
        server_display = Path(state.server_bin).name if state.server_bin else "(Auto)"
        env_display = state.env_overrides if state.env_overrides else "(None)"
        advanced_display = state.advanced_args if state.advanced_args else "(None)"

        tokens_display = f"{state.n_predict} / {state.max_tokens_list}"
        menu = [
            "--clear",
            "--backtitle",
            "Llama.cpp Local Test Launcher",
            "--title",
            "Main Menu",
            "--menu",
            "Select an option to configure or run:",
            "22",
            "70",
            "10",
            "1",
            f"Test:   {state.test_label}",
            "2",
            f"Model:  {model_display}",
            "3",
            f"Server: {server_display}",
            "4",
            f"Tokens: {tokens_display}",
            "5",
            f"Env:    {env_display}",
            "6",
            f"Advanced: {advanced_display}",
            "7",
            "RUN SELECTED TEST",
            "8",
            "CONFIGURE AND RUN ROUND ROBIN",
            "9",
            "Exit",
        ]

        choice, code = run_dialog(menu)
        if code != 0:
            break

        if choice == "1":
            select_test(state)
        elif choice == "2":
            select_model(state)
        elif choice == "3":
            edit_server_bin(state)
        elif choice == "4":
            tokens_menu(state)
        elif choice == "5":
            edit_env_overrides(state)
        elif choice == "6":
            edit_advanced_args(state)
        elif choice == "7":
            run_selected(state)
        elif choice == "8":
            round_robin_menu(state)
        elif choice == "9":
            break

    subprocess.run(["clear"])
    sys.exit(0)


if __name__ == "__main__":
    check_dependencies()
    try:
        warn_if_missing_nginx()
        main_menu()
    except KeyboardInterrupt:
        subprocess.run(["clear"])
        sys.exit(0)
