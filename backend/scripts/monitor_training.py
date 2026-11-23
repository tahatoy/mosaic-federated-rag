#!/usr/bin/env python3
"""
MOSAIC-v2 Real-Time Training Monitor
====================================
Reads `ai/experiments/training_status.json` and renders a live terminal dashboard
with progress bar, loss, epoch, ETA, and GPU VRAM statistics.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE = PROJECT_ROOT / "ai" / "experiments" / "training_status.json"

def render_progress_bar(percent: float, width: int = 30) -> str:
    filled = int(width * (percent / 100))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:.1f}%"

def show_status():
    if not STATUS_FILE.exists():
        print("Eğitim durumu dosyası henüz oluşturulmadı (ai/experiments/training_status.json).")
        return

    try:
        with open(STATUS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Durum dosyası okunurken hata: {e}")
        return

    status = data.get("status", "bilinmiyor")
    step = data.get("step", 0)
    total_steps = data.get("total_steps", 0)
    percent = data.get("percent", 0.0)
    epoch = data.get("epoch", 0.0)
    total_epochs = data.get("total_epochs", 1)
    loss = data.get("current_loss", 0.0)
    avg_loss = data.get("avg_loss", 0.0)
    lr = data.get("learning_rate", "0")
    sps = data.get("steps_per_sec", 0.0)
    elapsed = data.get("elapsed_time", "0s")
    eta = data.get("eta", "Hesaplanıyor...")
    vram_alloc = data.get("gpu_memory_allocated_gb", 0.0)
    vram_res = data.get("gpu_memory_reserved_gb", 0.0)
    ts = data.get("timestamp", "")

    bar = render_progress_bar(percent, width=32)

    print("\n" + "=" * 65)
    print("       MOSAIC-v2 LLM CANLI EĞİTİM TAKİP PANELİ (RTX 5060 Ti)")
    print("=" * 65)
    print(f" Durum          : {status.upper()}")
    print(f" İlerleme       : {bar}")
    print(f" Adım (Step)    : {step:,} / {total_steps:,} (Epoch {epoch:.2f} / {total_epochs})")
    print(f" Anlık Kayıp    : {loss:.4f} (Ortalama: {avg_loss:.4f})")
    print(f" Öğrenme Oranı  : {lr}")
    print(f" Hız            : {sps:.2f} adım/sn")
    print(f" Geçen Süre     : {elapsed}")
    print(f" Kalan Süre     : {eta}")
    print(f" GPU Belleği    : {vram_alloc:.2f} GB ayrılan / {vram_res:.2f} GB rezerve (16 GB)")
    print(f" Son Güncelleme : {ts}")
    print("=" * 65)

    if status == "completed":
        print("\n [TAMAMLANDI] Eğitim başarıyla sonuçlandı ve model kaydedildi.")
        print(f"   Nihai Kayıp: {data.get('final_loss', 0.0)}")
        print(f"   Toplam Süre: {data.get('total_duration_human', '')}")
        print("   Değerlendirme Raporu: ai/experiments/eval_report_epoch1.md")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        try:
            while True:
                # Clear terminal
                print("\033[H\033[J", end="")
                show_status()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nTakip durduruldu.")
    else:
        show_status()
