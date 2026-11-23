"""
MOSAIC-v2 Unified Launcher
==========================
HQ (WSL) ve Branch VM ortamları için merkezi başlatıcı.
Ortamı otomatik tespit eder ve ilgili servisleri ayağa kaldırır.

Usage:
  python scripts/launcher.py --all            # Ortama göre her şeyi başlat
  python scripts/launcher.py --stop           # Tüm servisleri durdur
  python scripts/launcher.py --status         # Servis durumlarını göster
  python scripts/launcher.py --llm            # LLM (llama-server) başlat
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import json
import httpx
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ai"))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# .env veya .env.client dosyasını yükle
if os.path.exists(PROJECT_ROOT / ".env"):
    logger.info(f"Yapılandırma yüklendi: {PROJECT_ROOT / '.env'}")
    load_dotenv(PROJECT_ROOT / ".env")
elif os.path.exists(PROJECT_ROOT / ".env.client"):
    logger.info(f"Yapılandırma yüklendi: {PROJECT_ROOT / '.env.client'}")
    load_dotenv(PROJECT_ROOT / ".env.client")
else:
    logger.warning("Yapılandırma dosyası (.env veya .env.client) bulunamadı!")

import importlib.util
import tempfile

# Configuration
_temp_dir = Path("/tmp") if sys.platform != "win32" and os.path.exists("/tmp") else Path(tempfile.gettempdir())
PID_FILE_HQ = str(_temp_dir / "mosaic_pids.json")
PID_FILE_BRANCH = str(_temp_dir / "mosaic_branch_pids.json")

# Ports
ROUTER_PORT = int(os.getenv("ROUTER_PORT", "8000"))
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8004"))
UI_PORT = int(os.getenv("UI_PORT", "8005"))
FL_PORT = int(os.getenv("SERVER_FLOWER_PORT", "9080"))
LLM_GEMMA_PORT = 8099
LLM_QWEN_PORT = 8098

def detect_environment():
    """HQ mı branch mı olduğunu otomatik tespit eder."""
    env = os.getenv("MOSAIC_ENV", "").lower()
    if env in ("hq", "branch"):
        return env
    if os.getenv("BRANCH_ID"):
        return "branch"
    return "hq"

def is_pid_running(pid):
    """PID'nin çalışıp çalışmadığını platforma uygun şekilde kontrol eder."""
    try:
        pid = int(pid)
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            err = ctypes.windll.kernel32.GetLastError()
            if err == 5:  # ERROR_ACCESS_DENIED (çalışıyor demektir)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False

def free_ports(ports):
    """Belirtilen portları hem Linux hem Windows (WSL2 köprüsü dahil) tarafında temizler."""
    if not ports:
        return
    if sys.platform == "win32":
        for p in ports:
            try:
                out = subprocess.check_output(f"netstat -ano | findstr :{p}", shell=True, text=True)
                for line in out.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 5 and "LISTENING" in parts[3].upper():
                        pid = parts[-1]
                        if pid != "0" and int(pid) != os.getpid():
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    else:
        for p in ports:
            try:
                subprocess.run(["fuser", "-k", f"{p}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # Eğer WSL2 içindeysek, Windows tarafında portu meşgul eden süreç varsa onu da serbest bırak
        if os.path.exists("/mnt/c"):
            try:
                port_list = ",".join(str(p) for p in ports)
                ps_cmd = f"Get-NetTCPConnection -LocalPort {port_list} -State Listen -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"
                subprocess.run(["powershell.exe", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

def check_dependencies(is_branch: bool = False, skip: bool = False):
    if skip:
        logger.info("[MOSAIC] Bağımlılık kontrolü atlandı (--skip-dep-check)")
        return

    required = {
        "flwr": "flwr>=1.8.0",
        "sentence_transformers": "sentence-transformers>=2.7.0",
        "chromadb": "chromadb",
        "httpx": "httpx>=0.27.0",
        "dotenv": "python-dotenv>=1.0.0",
    }

    # HQ ortamında ek bağımlılıklar gereklidir
    if not is_branch:
        required["diffprivlib"] = "diffprivlib>=0.6.0"
        required["apscheduler"] = "apscheduler>=3.10.0"

    missing = []
    for module, package in required.items():
        try:
            # Use importlib.util.find_spec to check without heavy runtime side effects
            spec = importlib.util.find_spec(module)
            if spec is None:
                missing.append(package)
        except Exception:
            missing.append(package)

    if missing:
        print("[HATA] Eksik bağımlılıklar tespit edildi:")
        for pkg in missing:
            print(f"  pip install {pkg}")
        req_file = "requirements_client.txt" if is_branch else "requirements.txt"
        print(f"\nTümünü kurmak için: pip install -r {req_file}")
        sys.exit(1)
    logger.info(f"[MOSAIC] Tüm bağımlılıklar mevcut ({'Branch' if is_branch else 'HQ'}) [OK]")

def get_pid_file():
    return PID_FILE_HQ if detect_environment() == "hq" else PID_FILE_BRANCH

def save_pids(pids):
    pid_file = get_pid_file()
    with open(pid_file, "w") as f:
        json.dump(pids, f)

def load_pids():
    pid_file = get_pid_file()
    if os.path.exists(pid_file):
        with open(pid_file, "r") as f:
            return json.load(f)
    return {}

def wait_for_service(name, url, timeout=120):
    start_time = time.time()
    logger.info(f"[{name}] Bekleniyor: {url} (max {timeout}s) ...")
    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=2.0, trust_env=False) as client:
                resp = client.get(url)
                if resp.status_code == 200:
                    logger.success(f"[{name}] Hazır [OK]")
                    return True
        except Exception:
            pass
        time.sleep(2)
    logger.error(f"[{name}] {timeout} saniyede ayağa kalkmadı — devam ediliyor")
    return False

def start_process(name, cmd, cwd, env_extra=None, log_file=None):
    logger.info(f"Başlatılıyor: {name} ...")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    py_path = env.get("PYTHONPATH", "")
    sep = ";" if sys.platform == "win32" else ":"
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{sep}{PROJECT_ROOT}/ai{sep}{PROJECT_ROOT}/backend{sep}{py_path}" if py_path else f"{PROJECT_ROOT}{sep}{PROJECT_ROOT}/ai{sep}{PROJECT_ROOT}/backend"
    env["PYTHONUNBUFFERED"] = "1"

    if sys.platform != "win32":
        lib_dir = str(PROJECT_ROOT / "ai" / "llama.cpp" / "build" / "bin")
        ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir

    if not log_file:
        log_file = str(_temp_dir / f"mosaic_{name.lower().replace(' ', '_')}.log")

    f = open(log_file, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(sys.platform != "win32")
    )
    return proc

def run_hq(args):
    logger.info("[MOSAIC] Ortam tespit edildi: HQ (WSL)")
    logger.info("[MOSAIC] Ortam tespit edildi: HQ (WSL/Windows)")
    free_ports([ROUTER_PORT, ORCHESTRATOR_PORT, UI_PORT, FL_PORT])
    pids = {}

    # 1. Router
    if not args.no_router:
        proc = start_process(
            "Router",
            [sys.executable, "backend/backend_api.py", "--service", "router", "--port", str(ROUTER_PORT)],
            PROJECT_ROOT,
            {"PORT": str(ROUTER_PORT)}
        )
        pids["Router"] = proc.pid
        wait_for_service("Router", f"http://127.0.0.1:{ROUTER_PORT}/health")

    # 2. RAG Services
    if not args.no_rag:
        # Ports for local RAGs
        rag_p1 = int(os.getenv("BRANCH1_RAG_PORT", "8001"))
        rag_p2 = int(os.getenv("BRANCH2_RAG_PORT", "8002"))
        rag_phq = int(os.getenv("HQ_RAG_PORT", "8003"))

        # Branch-1 RAG (Sadece lokal adrese işaret ediyorsa başlat)
        b1_url = os.getenv("BRANCH1_RAG_URL", "")
        b1_is_local = not b1_url or "localhost" in b1_url or "127.0.0.1" in b1_url
        if b1_is_local:
            proc1 = start_process(
                "Branch-1 RAG",
                [sys.executable, "backend/backend_api.py", "--service", "rag", "--port", str(rag_p1)],
                PROJECT_ROOT,
                {"BRANCH_ID": "branch_01", "PORT": str(rag_p1)}
            )
            pids["Branch-1 RAG"] = proc1.pid
        else:
            logger.info(f"Branch-1 RAG uzak adreste tanımlı ({b1_url}), lokal başlatma atlanıyor.")

        # Branch-2 RAG (Sadece lokal adrese işaret ediyorsa başlat)
        b2_url = os.getenv("BRANCH2_RAG_URL", "")
        b2_is_local = not b2_url or "localhost" in b2_url or "127.0.0.1" in b2_url
        if b2_is_local:
            proc2 = start_process(
                "Branch-2 RAG",
                [sys.executable, "backend/backend_api.py", "--service", "rag", "--port", str(rag_p2)],
                PROJECT_ROOT,
                {"BRANCH_ID": "branch_02", "PORT": str(rag_p2)}
            )
            pids["Branch-2 RAG"] = proc2.pid
        else:
            logger.info(f"Branch-2 RAG uzak adreste tanımlı ({b2_url}), lokal başlatma atlanıyor.")

        # HQ RAG
        prochq = start_process(
            "HQ RAG",
            [sys.executable, "backend/backend_api.py", "--service", "rag", "--port", str(rag_phq)],
            PROJECT_ROOT,
            {"BRANCH_ID": "hq", "PORT": str(rag_phq)}
        )
        pids["HQ RAG"] = prochq.pid

        if b1_is_local:
            wait_for_service("Branch-1 RAG", f"http://127.0.0.1:{rag_p1}/health")
        if b2_is_local:
            wait_for_service("Branch-2 RAG", f"http://127.0.0.1:{rag_p2}/health")
        wait_for_service("HQ RAG", f"http://127.0.0.1:{rag_phq}/health")

    # 3. Orchestrator
    if not args.no_orchestrator:
        proc = start_process(
            "Orchestrator",
            [sys.executable, "backend/backend_api.py", "--service", "orchestrator", "--port", str(ORCHESTRATOR_PORT)],
            PROJECT_ROOT,
            {"PORT": str(ORCHESTRATOR_PORT)}
        )
        pids["Orchestrator"] = proc.pid
        wait_for_service("Orchestrator", f"http://127.0.0.1:{ORCHESTRATOR_PORT}/health")

    # 4. Lightweight Web UI
    if not args.no_ui:
        proc = start_process(
            "UI",
            [sys.executable, "-m", "uvicorn", "frontend.web_server:app", "--host", "0.0.0.0", "--port", str(UI_PORT)],
            PROJECT_ROOT
        )
        pids["UI"] = proc.pid
        time.sleep(2)

    # 5. Flower Server
    if not args.no_flower:
        proc = start_process(
            "Flower Server",
            [sys.executable, "backend/federated/server/server.py"],
            PROJECT_ROOT,
            {"FL_DAEMON": "true"}
        )
        pids["Flower Server"] = proc.pid
        time.sleep(5)

    # 6. LLM Servisleri
    if args.llm or args.all:
        logger.info("LLM servisleri başlatılıyor (Gemma + Qwen)...")

        # Binary tespiti
        llama_bin = PROJECT_ROOT / "ai" / "llama.cpp" / "build" / "bin" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
        if not llama_bin.exists():
            llama_bin = "llama-server"
        elif sys.platform != "win32":
            try:
                import stat
                st = os.stat(llama_bin)
                os.chmod(llama_bin, st.st_mode | stat.S_IEXEC)
            except Exception as e:
                logger.warning(f"llama-server çalıştırılabilirlik izni (chmod +x) eklenemedi: {e}")

        # Gemma (Simple/Chitchat)
        gemma_path = PROJECT_ROOT / "ai" / "models" / "quantized" / "gemma-3-4b-Q4_K_M.gguf"
        if gemma_path.exists():
            proc_g = start_process("LLM_Gemma", [str(llama_bin), "-m", str(gemma_path), "--port", str(LLM_GEMMA_PORT), "-ngl", "99", "-c", "4096", "-np", "1"], PROJECT_ROOT)
            pids["LLM Gemma"] = proc_g.pid
            logger.success(f"LLM Gemma başlatıldı (Port: {LLM_GEMMA_PORT})")
        else:
            logger.error(f"Gemma bulunamadı: {gemma_path}")

        # Qwen (Complex)
        qwen_path = PROJECT_ROOT / "ai" / "models" / "finetuned" / "QWEN3_EXPERT_V2.Q4_K_M.gguf"
        lora_path = PROJECT_ROOT / "ai" / "models" / "finetuned" / "qwen3_lora_finance.gguf"
        use_lora = os.getenv("USE_QWEN_LORA", "false").lower() in ("1", "true", "yes")
        if qwen_path.exists():
            cmd_qwen = [str(llama_bin), "-m", str(qwen_path), "--port", str(LLM_QWEN_PORT), "-ngl", "99", "-c", "4096", "-np", "1"]
            if use_lora and lora_path.exists():
                cmd_qwen.extend(["--lora", str(lora_path)])
                logger.info(f"Qwen Finans LoRA adaptörü eklendi: {lora_path.name} (349 MB)")
            else:
                logger.info(f"Qwen ana uzman model devrede: {qwen_path.name}")
            proc_q = start_process("LLM_Qwen", cmd_qwen, PROJECT_ROOT)
            pids["LLM Qwen"] = proc_q.pid
            logger.success(f"LLM Qwen başlatıldı (Port: {LLM_QWEN_PORT})")
        else:
            logger.error(f"Qwen bulunamadı: {qwen_path}")

    save_pids(pids)
    logger.success("[MOSAIC] HQ servisleri başlatıldı.")

def run_branch(args):
    if getattr(args, "branch1", False):
        branch_id = "branch_01"
        port = 8001
    elif getattr(args, "branch2", False):
        branch_id = "branch_02"
        port = 8002
    else:
        branch_id = os.getenv("BRANCH_ID", "branch_01")
        default_port = "8002" if ("02" in branch_id or "2" in branch_id) else "8001"
        port = int(os.getenv("PORT", os.getenv("BRANCH_RAG_PORT", default_port)))

    os.environ["BRANCH_ID"] = branch_id
    os.environ["PORT"] = str(port)
    os.environ["BRANCH_RAG_PORT"] = str(port)
    os.environ["MOSAIC_ENV"] = "branch"

    # Port çakışmasını önlemek için eski süreç varsa sonlandır
    if sys.platform != "win32":
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    logger.info(f"[MOSAIC] Ortam: Branch (BRANCH_ID={branch_id}, RAG_PORT={port})")
    pids = {}

    # 1. Branch RAG API
    rag_log = str(_temp_dir / f"mosaic_rag_{branch_id}.log")
    proc = start_process(
        "Branch RAG",
        [sys.executable, "backend/backend_api.py", "--service", "rag", "--port", str(port)],
        PROJECT_ROOT,
        {"BRANCH_ID": branch_id, "PORT": str(port), "BRANCH_RAG_PORT": str(port)},
        log_file=rag_log
    )
    pids["Branch RAG"] = proc.pid

    ready = wait_for_service("Branch RAG", f"http://127.0.0.1:{port}/health", timeout=60)
    if not ready and os.path.exists(rag_log):
        logger.error(f"[Branch RAG] Başlatma günlüğü ({rag_log}) son 15 satır:")
        try:
            with open(rag_log, "r", encoding="utf-8", errors="replace") as lf:
                lines = lf.readlines()
                for line in lines[-15:]:
                    print("  |", line.rstrip())
        except Exception:
            pass

    # 2. Branch UI
    if not args.no_ui:
        proc = start_process(
            "Branch UI",
            [sys.executable, "-m", "uvicorn", "frontend.web_server:app", "--host", "0.0.0.0", "--port", str(UI_PORT)],
            PROJECT_ROOT
        )
        pids["Branch UI"] = proc.pid
        time.sleep(2)

    # 3. Flower Clients
    if not args.no_flower:
        # Always start Flower clients in branch mode (they poll the server in a daemon loop)
        logger.info("[FL] Flower istemcileri başlatılıyor (Daemon modunda sunucu bekleniyor)...")
        # Flower LLM
        proc_llm = start_process(
            "Flower LLM",
            [sys.executable, "backend/client_setup/flower_rag_client.py", "--mode", "llm"],
            PROJECT_ROOT
        )
        pids["Flower LLM"] = proc_llm.pid
        time.sleep(2)

        # Flower Embedding
        proc_emb = start_process(
            "Flower Embedding",
            [sys.executable, "backend/client_setup/flower_rag_client.py", "--mode", "embedding"],
            PROJECT_ROOT
        )
        pids["Flower Embedding"] = proc_emb.pid

    save_pids(pids)
    logger.success(f"[MOSAIC] {branch_id} servisleri başlatıldı.")

def stop_all():
    pids = load_pids()
    if not pids:
        logger.info("Durdurulacak aktif process bulunamadı.")
        return

    for name, pid in pids.items():
        logger.info(f"Durduruluyor: {name} (PID: {pid})")
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError, SystemError):
            continue

    time.sleep(3)
    # Re-check and SIGKILL if necessary
    for name, pid in pids.items():
        if is_pid_running(pid):
            logger.warning(f"{name} hala çalışıyor, SIGKILL gönderiliyor...")
            try:
                if sys.platform == "win32":
                    os.kill(int(pid), signal.SIGTERM)
                else:
                    os.kill(int(pid), signal.SIGKILL)
            except Exception:
                pass

    pid_file = get_pid_file()
    if os.path.exists(pid_file):
        os.remove(pid_file)
    logger.success("Tüm servisler durduruldu.")

def show_status():
    pids = load_pids()
    env = detect_environment()

    print(f"\n[MOSAIC] Ortam: {env.upper()}")
    print(f"{'Servis':<20} {'Port':<8} {'Durum':<10} {'PID':<8}")
    print("-" * 50)

    services = []
    if env == "hq":
        b1_url = os.getenv("BRANCH1_RAG_URL", f"http://127.0.0.1:{os.getenv('BRANCH1_RAG_PORT', '8001')}")
        b2_url = os.getenv("BRANCH2_RAG_URL", f"http://127.0.0.1:{os.getenv('BRANCH2_RAG_PORT', '8002')}")
        services = [
            ("Router", ROUTER_PORT, f"http://127.0.0.1:{ROUTER_PORT}/health"),
            ("Branch-1 RAG", int(os.getenv("BRANCH1_RAG_PORT", "8001")), f"{b1_url.rstrip('/')}/health"),
            ("Branch-2 RAG", int(os.getenv("BRANCH2_RAG_PORT", "8002")), f"{b2_url.rstrip('/')}/health"),
            ("HQ RAG", int(os.getenv("HQ_RAG_PORT", "8003")), f"http://127.0.0.1:{os.getenv('HQ_RAG_PORT', '8003')}/health"),
            ("Orchestrator", ORCHESTRATOR_PORT, f"http://127.0.0.1:{ORCHESTRATOR_PORT}/health"),
            ("UI", UI_PORT, None),
            ("Flower Server", FL_PORT, None),
        ]
    else:
        branch_id = os.getenv("BRANCH_ID", "branch_01")
        default_port = "8002" if ("02" in branch_id or "2" in branch_id) else "8001"
        port = int(os.getenv("PORT", os.getenv("BRANCH_RAG_PORT", default_port)))
        services = [
            ("Branch RAG", port, f"http://127.0.0.1:{port}/health"),
            ("Branch UI", UI_PORT, None),
            ("Flower LLM", None, None),
            ("Flower Embedding", None, None),
        ]

    for name, port, health_url in services:
        status = "DOWN"
        pid = pids.get(name, "-")
        reason = ""

        # 1. Sağlık kontrolü
        if health_url:
            try:
                with httpx.Client(timeout=4.0, trust_env=False) as client:
                    r = client.get(health_url)
                    if r.status_code == 200:
                        try:
                            data = r.json()
                            if data.get("status") == "degraded":
                                status = "DEGRADED"
                            else:
                                status = "UP"
                        except Exception:
                            status = "UP"
                    else:
                        status = f"HTTP {r.status_code}"
            except httpx.ConnectTimeout:
                reason = "(Timeout)"
            except httpx.ConnectError:
                reason = "(Bağlanamadı)"
            except Exception as e:
                reason = f"({type(e).__name__})"
        elif port:
            import socket
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    status = "RUNNING"
            except Exception:
                pass

        # 2. Servis yanıt vermiyorsa PID durumuna bak
        if status == "DOWN":
            if pid != "-" and not is_pid_running(pid):
                status = "CRASHED"
            elif pid != "-" and is_pid_running(pid):
                status = "STARTING"
            elif reason:
                status = f"DOWN {reason}"

        print(f"{name:<20} {str(port) if port else '-':<8} {status:<15} {str(pid):<8}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOSAIC-v2 Unified Launcher")
    parser.add_argument("--all", action="store_true", help="Tüm servisleri başlat")
    parser.add_argument("--stop", action="store_true", help="Tüm çalışan servisleri durdur")
    parser.add_argument("--status", action="store_true", help="Servis sağlık durumlarını göster")
    parser.add_argument("--llm", action="store_true", help="LLM sunucularını başlat (Gemma/Qwen)")
    parser.add_argument("--hq", action="store_true", help="Genel Merkez (HQ) modunda başlat")
    parser.add_argument("--branch", action="store_true", help="Şube (VM) modunda başlat")
    parser.add_argument("--branch1", action="store_true", help="Şube-1 (Port 8001) olarak başlat")
    parser.add_argument("--branch2", action="store_true", help="Şube-2 (Port 8002) olarak başlat")
    parser.add_argument("--skip-dep-check", action="store_true", help="Bağımlılık kontrolünü atla")
    parser.add_argument("--no-flower", action="store_true", help="Flower federasyonunu başlatma")
    parser.add_argument("--no-ui", action="store_true", help="Web arayüzünü başlatma")
    parser.add_argument("--no-router", action="store_true", help="Router servisini başlatma")
    parser.add_argument("--no-rag", action="store_true", help="RAG servisini başlatma")
    parser.add_argument("--no-orchestrator", action="store_true", help="Orchestrator servisini başlatma")
    args = parser.parse_args()

    # Ortam değişkeni önceliklendirmesi
    if args.branch1:
        os.environ["MOSAIC_ENV"] = "branch"
        os.environ["BRANCH_ID"] = "branch_01"
        os.environ["BRANCH_RAG_PORT"] = "8001"
    elif args.branch2:
        os.environ["MOSAIC_ENV"] = "branch"
        os.environ["BRANCH_ID"] = "branch_02"
        os.environ["BRANCH_RAG_PORT"] = "8002"
    elif args.branch:
        os.environ["MOSAIC_ENV"] = "branch"
    elif args.hq:
        os.environ["MOSAIC_ENV"] = "hq"

    if args.stop:
        stop_all()
        sys.exit(0)

    if args.status:
        show_status()
        sys.exit(0)

    env = detect_environment()
    is_branch = (env == "branch")
    check_dependencies(is_branch=is_branch, skip=args.skip_dep_check)

    if args.all or args.llm:
        if env == "hq":
            run_hq(args)
        else:
            run_branch(args)

        if sys.platform == "win32":
            logger.info("Windows platform detected. Keeping launcher alive to monitor child processes. Press Ctrl+C to stop all.")
            try:
                while True:
                    time.sleep(2)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Stopping all services...")
                stop_all()
    else:
        parser.print_help()
