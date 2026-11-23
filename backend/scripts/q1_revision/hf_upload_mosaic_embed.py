"""Upload MOSAIC-Embed (ai/models/embeddings/turkembed_v3) and its model card to the Hugging Face Hub.

Usage: python hf_upload_mosaic_embed.py <model_card.md> [repo_id]   (requires a cached `huggingface-cli login`)
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "ai" / "models" / "embeddings" / "turkembed_v3"
CARD = Path(sys.argv[1])
REPO = sys.argv[2] if len(sys.argv) > 2 else "Toxotes/mosaic-embed"

api = HfApi()
api.whoami()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path=str(MODEL), repo_id=REPO, ignore_patterns=["README.md", "checkpoint*"],
                  commit_message="MOSAIC-Embed: multilingual-e5-small fine-tuned on query-to-passage pairs")
api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md", repo_id=REPO, commit_message="Model card")
print("https://huggingface.co/" + REPO)
