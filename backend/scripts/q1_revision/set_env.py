"""Bir .env dosyasında anahtarları ayarlar (yoksa ekler). Kullanım: python set_env.py <dosya> KEY=VAL ..."""
import re, sys
p = sys.argv[1]; s = open(p, encoding="utf-8").read()
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    if re.search(rf"^{re.escape(k)}=", s, re.M):
        s = re.sub(rf"^{re.escape(k)}=.*$", f"{k}={v}", s, flags=re.M)
    else:
        s = s.rstrip("\n") + f"\n{k}={v}\n"
open(p, "w", encoding="utf-8").write(s)
print("ayarlandı:", p, sys.argv[2:])
