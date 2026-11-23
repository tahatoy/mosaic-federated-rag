#!/usr/bin/env python3
"""
MOSAIC-v2: Hugging Face Hub Model Upload Tool
==============================================
Uploads fine-tuned and quantized GGUF models and documentation to Hugging Face Hub:
  - Qwen3-8B  -> Toxotes/QWEN3_EXPERT_V2_GGUF
  - Gemma-3-4B -> Toxotes/GEMMA3_FINANCE_V1_GGUF

Usage:
  python backend/scripts/upload_to_hf.py --dry-run
  python backend/scripts/upload_to_hf.py --model qwen
  python backend/scripts/upload_to_hf.py --model gemma
  python backend/scripts/upload_to_hf.py --model all
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, get_token
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "qwen": {
        "repo_id": "Toxotes/QWEN3_EXPERT_V2_GGUF",
        "description": "Mosaic-Qwen (Qwen3-8B Turkish Finance Expert, Q4_K_M GGUF)",
        "readme": PROJECT_ROOT / "ai" / "models" / "finetuned" / "QWEN3_EXPERT_V2_GGUF_README.md",
        "files": [
            PROJECT_ROOT / "ai" / "models" / "finetuned" / "QWEN3_EXPERT_V2.Q4_K_M.gguf"
        ]
    },
    "gemma": {
        "repo_id": "Toxotes/GEMMA3_FINANCE_V1_GGUF",
        "description": "GEMMA3-FINANCE-V1 (Gemma-3-4B Turkish Finance Edge SLM, Q4_K_M GGUF)",
        "readme": PROJECT_ROOT / "ai" / "models" / "finetuned" / "GEMMA3_FINANCE_V1_GGUF_README.md",
        "files": [
            PROJECT_ROOT / "ai" / "models" / "finetuned" / "gemma-3-4b-finance-lora.gguf",
            PROJECT_ROOT / "ai" / "models" / "finetuned" / "GEMMA3_FINANCE_V1.Q4_K_M.gguf"
        ]
    }
}

def upload_model(api: HfApi, model_key: str, dry_run: bool = False):
    cfg = MODELS[model_key]
    repo_id = cfg["repo_id"]
    desc = cfg.get("description", model_key)

    logger.info("=" * 65)
    logger.info(f"Model: {desc}")
    logger.info(f"Target Repository: https://huggingface.co/{repo_id}")
    logger.info("=" * 65)

    if dry_run:
        logger.info("[DRY-RUN] Modu aktif. Gerçek dosya transferi yapılmayacak.")
    else:
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
            logger.info(f"Repository doğrulandı: {repo_id}")
        except Exception as e:
            logger.error(f"Repository oluşturulamadı/erişilemedi: {e}")
            return False

    # 1. Upload README if present
    readme_path = cfg["readme"]
    if readme_path.exists():
        size_kb = readme_path.stat().st_size / 1024
        logger.info(f"README: {readme_path.name} ({size_kb:.1f} KB)")
        if not dry_run:
            try:
                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model"
                )
                logger.info("[OK] README başarıyla yüklendi.")
            except Exception as e:
                logger.error(f"README yükleme hatası: {e}")
                return False
    else:
        logger.warning(f"README bulunamadı: {readme_path}")

    # 2. Upload model files
    for file_path in cfg["files"]:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            size_gb = size_mb / 1024
            size_str = f"{size_gb:.2f} GB" if size_mb >= 1024 else f"{size_mb:.2f} MB"
            logger.info(f"Dosya: {file_path.name} ({size_str})")

            if not dry_run:
                try:
                    logger.info(f"-> Yükleme başlatılıyor: {file_path.name}...")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    logger.info(f"[OK] Başarıyla yüklendi: {file_path.name}")
                except Exception as e:
                    logger.error(f"Dosya yükleme hatası ({file_path.name}): {e}")
                    logger.info("Büyük dosyalarda ağ kesintisi olduysa komutu tekrar çalıştırabilirsiniz.")
                    return False
        else:
            logger.error(f"Kritik model dosyası bulunamadı: {file_path}")
            return False

    status_tag = "[DRY-RUN TAMAMLANDI]" if dry_run else "[YÜKLEME BAŞARILI]"
    logger.info(f"{status_tag} {model_key.upper()} -> https://huggingface.co/{repo_id}\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload MOSAIC-v2 GGUF models to Hugging Face Hub")
    parser.add_argument(
        "--model",
        choices=["qwen", "gemma", "all"],
        default="all",
        help="Yüklenecek model seçimi (varsayılan: all)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (varsayılan: cache veya HF_TOKEN env)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dosyaları yüklemeden varlıklarını, boyutlarını ve hedef repoları kontrol eder"
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
    if not token and not args.dry_run:
        logger.error("Hugging Face API token bulunamadı! 'huggingface-cli login' çalıştırın veya --token parametresi verin.")
        sys.exit(1)

    api = HfApi(token=token) if token else None

    # Verify authentication in non-dry-run mode
    if api and not args.dry_run:
        try:
            user_info = api.whoami()
            logger.info(f"Hugging Face Girişi Doğrulandı: {user_info.get('name', 'Bilinmeyen Kullanıcı')}")
        except Exception as e:
            logger.error(f"Token doğrulanamadı: {e}")
            sys.exit(1)

    success = True
    if args.model in ("qwen", "all"):
        if not upload_model(api, "qwen", dry_run=args.dry_run):
            success = False
    if args.model in ("gemma", "all"):
        if not upload_model(api, "gemma", dry_run=args.dry_run):
            success = False

    if not success:
        logger.error("Bazı dosyaların yüklenmesinde hata oluştu.")
        sys.exit(1)
    else:
        logger.info("Tüm işlemler başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
    main()
