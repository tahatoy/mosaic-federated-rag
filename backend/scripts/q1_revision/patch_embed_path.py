"""ai_engine._embedding_model_path'e EMBEDDING_MODEL_PATH ortam değişkeni desteği ekler (federe MOSAIC-Embed'i canlıya almak için).
Değişken boşsa davranış değişmez. Kullanım: python patch_embed_path.py backend/ai_engine.py"""
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
if "EMBEDDING_MODEL_PATH" in s:
    print("zaten yamalı:", p); sys.exit(0)
a = '    base = PROJECT_ROOT / "ai" / "models" / "embeddings"\n    if key == "mosaic_embed":'
b = ('    override = os.getenv("EMBEDDING_MODEL_PATH", "").strip()\n'
     '    if override:  # ör. Flower ile federe eğitilmiş MOSAIC-Embed (ai/models/embeddings/mosaic_embed_flower_vm)\n'
     '        op = Path(override)\n'
     '        return op if op.is_absolute() else PROJECT_ROOT / op\n' + a)
crlf = "\r\n" in s
s2 = s.replace("\r\n", "\n")
if a not in s2:
    print("HATA: beklenen satır bulunamadı:", p); sys.exit(2)
s2 = s2.replace(a, b, 1)
open(p, "w", encoding="utf-8", newline="\r\n" if crlf else "\n").write(s2)
print("yamalandı:", p)
