"""
MOSAIC-v2 System & GPU Tool
===========================
Centralized tool for GPU configuration, LLM server management, and system diagnostics.

Combines:
  - _gpu_setup.py (Central GPU config)
  - _llm_server.py (llama-server launcher)
  - _test_llm.py (GPU/LLM diagnostics)

Usage:
  python scripts/sys_tool.py report               # GPU & System report
  python scripts/sys_tool.py test [--quick]       # Run diagnostics
  python scripts/sys_tool.py start [--model gemma] # Start llama-server
  python scripts/sys_tool.py stop                # Stop llama-server
  python scripts/sys_tool.py status              # Check llama-server status
  python scripts/sys_tool.py network             # Note on Windows network setup
"""

import os
import sys
import shutil
import urllib.request
import urllib.error
import json
import subprocess
import time
import signal
import argparse
from pathlib import Path

# --- Platform Detection ---
IS_WINDOWS = os.name == "nt"
_WSL_DISTRO = os.environ.get("MOSAIC_WSL_DISTRO", "Ubuntu")
_WSL_UNC    = Path(f"//wsl$/{_WSL_DISTRO}")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PID_FILE = Path("/tmp/llama_server.pid")
DEFAULT_PORT = 8099

# --- Path Helpers ---
def _wsl_path(linux_path: str) -> Path:
    if IS_WINDOWS:
        try:
            return _WSL_UNC / linux_path.lstrip("/")
        except Exception:
            return Path(linux_path)
    return Path(linux_path)

WSL_GGUF_DIR = _wsl_path(os.environ.get("GGUF_DIR", "/opt/mosaic/gguf"))
LLM_SERVER_URL = os.environ.get("LLM_SERVER_URL", f"http://127.0.0.1:{DEFAULT_PORT}")

def _find_llama_server() -> Path | None:
    custom = os.environ.get("LLAMA_SERVER_BIN")
    if custom and Path(custom).exists():
        return Path(custom)

    wsl_build = Path(__file__).resolve().parents[2] / "ai" / "llama.cpp" / "build" / "bin" / "llama-server"
    try:
        if wsl_build.exists():
            return wsl_build
    except (PermissionError, OSError):
        pass

    found = shutil.which("llama-server")
    if found:
        p = Path(found)
        if IS_WINDOWS and p.suffix.upper() == ".EXE":
            return p
        return p

    fallback = PROJECT_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
    if fallback.exists():
        return fallback
    return None

LLAMA_SERVER_BIN = _find_llama_server()

GGUF_MODELS = {
    "gemma": WSL_GGUF_DIR / "gemma-3-4b-it-Q4_K_M.gguf",
    "qwen":  WSL_GGUF_DIR / "qwen3-8b-finance-Q4_K_M.gguf",
}

# --- GPU Config & Models ---
def torch_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None

def server_available(url: str = None) -> bool:
    u = url or LLM_SERVER_URL
    try:
        with urllib.request.urlopen(f"{u}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def llm_generate(prompt: str, max_tokens: int = 512, url: str = None) -> str:
    u = url or LLM_SERVER_URL
    if not server_available(u):
        raise RuntimeError(f"llama-server not responding at {u}")
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": max_tokens,
        "stream": False,
        "stop": ["<|im_end|>", "<end_of_turn>", "<|endoftext|>"]
    }).encode("utf-8")
    req = urllib.request.Request(f"{u}/completion", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r).get("content", "").strip()

# --- Commands ---
def start_server(model_key: str = "gemma", port: int = DEFAULT_PORT):
    if server_available(f"http://127.0.0.1:{port}"):
        print(f"llama-server already running on port {port}")
        return

    if not LLAMA_SERVER_BIN:
        print("ERROR: llama-server binary not found!")
        sys.exit(1)

    gguf = GGUF_MODELS.get(model_key)
    if not gguf or not gguf.exists():
        if model_key == "qwen" and GGUF_MODELS["gemma"].exists():
            print("Qwen not found, falling back to Gemma...")
            gguf = GGUF_MODELS["gemma"]
        else:
            print(f"GGUF not found: {gguf}")
            sys.exit(1)

    cmd = [
        str(LLAMA_SERVER_BIN),
        "--model", str(gguf),
        "--n-gpu-layers", "-1",
        "--ctx-size", "4096",
        "--batch-size", "512",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--threads", "4",
    ]
    print(f"Starting {model_key} on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    PID_FILE.write_text(str(proc.pid))

    for i in range(30):
        time.sleep(2)
        if server_available(f"http://127.0.0.1:{port}"):
            print(f"llama-server ready (PID: {proc.pid})")
            return
        print(f"Waiting... {i*2}s", end="\r")
    print("\nWarning: server took too long to start.")

def stop_server():
    if not PID_FILE.exists():
        print("No PID file found. Server might not be running.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
        PID_FILE.unlink(missing_ok=True)
        print(f"Stopped server (PID: {pid})")
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("Server was already stopped.")

def gpu_report():
    print("\n" + "="*60)
    print("GPU & SYSTEM REPORT")
    print("="*60)
    try:
        import torch
        print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"Device: {p.name} | VRAM: {p.total_memory//1024**2} MB")
    except ImportError:
        print("PyTorch: Not installed")

    print(f"llama-server binary: {LLAMA_SERVER_BIN or 'Not found'}")
    if server_available():
        print(f"llama-server: Running at {LLM_SERVER_URL}")
    else:
        print("llama-server: Not running")

    for k, v in GGUF_MODELS.items():
        print(f"GGUF [{k}]: {'Exists' if v.exists() else 'Missing'} ({v})")
    print("="*60 + "\n")

def run_tests(quick=False):
    print("\nRunning Diagnostics...")
    gpu_report()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    # Simple tensor test
    t = torch.randn(10, 10).cuda()
    _ = t @ t
    print("[OK] GPU Tensor operation successful")

    if server_available():
        try:
            resp = llm_generate("Hi!")
            print(f"[OK] LLM Inference successful: {resp[:50]}...")
        except Exception as e:
            print(f"[FAIL] LLM Inference failed: {e}")
    else:
        print("! llama-server not running, skipping inference test")

    if not quick:
        print("Running full model load tests...")
        # Add more tests here if needed (transformers, sentence-transformers)
        pass

def network_info():
    print("\nWindows Network Setup Note:")
    print("If you are running in WSL and need to access Windows-hosted services (or vice-versa),")
    print("ensure you've run the 'scripts/setup_windows_network.ps1' in an Administrator PowerShell.")
    print("This script configures firewall rules and port forwarding for the project's microservices.")
    ps_path = PROJECT_ROOT / "scripts" / "setup_windows_network.ps1"
    if ps_path.exists():
        print(f"Script location: {ps_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOSAIC-v2 System Tool")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("report")
    t = sub.add_parser("test")
    t.add_argument("--quick", action="store_true")
    s = sub.add_parser("start")
    s.add_argument("--model", default="gemma", choices=list(GGUF_MODELS))
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("network")

    args = parser.parse_args()
    if args.cmd == "report": gpu_report()
    elif args.cmd == "test": run_tests(args.quick)
    elif args.cmd == "start": start_server(args.model, args.port)
    elif args.cmd == "stop": stop_server()
    elif args.cmd == "status":
        if server_available(): print(f"Running at {LLM_SERVER_URL}")
        else: print("Not running")
    elif args.cmd == "network": network_info()
    else: parser.print_help()
