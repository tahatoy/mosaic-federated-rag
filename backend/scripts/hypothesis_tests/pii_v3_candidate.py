import os
import re

from loguru import logger

# Yaygın Türkçe ad sözlüğü (TÜİK en sık verilen adlar ve genel kullanım; test listelerinden bağımsız derlenmiştir)
_PII_FIRST_NAMES = set("""
ahmet mehmet mustafa ali hüseyin hasan ibrahim ismail osman yusuf murat ömer ramazan halil süleyman abdullah mahmut
recep salih fatih kadir emre hakan adem kemal yaşar bayram mustafa orhan cem can burak serkan erkan volkan onur kaan
kerem barış uğur tolga yasin enes furkan berk eren efe arda emir mert umut caner deniz ozan sinan tuncay tayfun
cengiz levent selim serdar sedat erdem gökhan alper koray tarık metin necati nihat ercan engin oğuz oğuzhan batuhan
yiğit kağan alperen emirhan berkay doruk görkem ilker kubilay erhan halit şükrü şahin taner turgut zafer
fatma ayşe emine hatice zeynep elif meryem şerife zehra sultan hanife merve özlem esra büşra kübra tuğba gizem
derya selin pınar ebru sevgi gül nur irem ece ceren melis buse damla dilek aslı aysel ayten nermin songül sibel
seda serap gülay gülşen hülya filiz yasemin yeliz canan esin şeyma rabia sena beyza nisa eylül zeynep defne ela
azra ecrin asya ilayda melike sude aleyna tuba hande gamze burcu didem özge çağla başak berna nazlı
""".split())
_PII_NOT_SURNAME = {"Bayramı", "Bayram", "Pasifik", "Sultan", "Han", "Paşa", "Camii", "Caddesi", "Sokak", "Bulvarı",
                    "Mahallesi", "Üniversitesi", "Hastanesi", "Köprüsü", "Holding", "Bank", "Bankası", "Grubu"}

_PII_NOT_SURNAME_UP = {"TL", "USD", "EUR", "TCMB", "BDDK", "SPK", "KAP", "BIST", "KKDF", "BSMV", "IBAN", "SWIFT", "EFT", "FAST", "ATM", "POS",
                       "A.Ş", "AŞ", "LTD", "ŞTİ", "TÜFE", "ÜFE", "GSYH", "KDV", "ÖTV", "SGK", "MASAK", "KVKK", "TBB", "BES", "TEFAS", "NAV", "CEO", "CFO"}

def _pii_lower(w):
    return w.replace("İ", "i").replace("I", "ı").lower()

# Türkçe banka belgelerinde kişi adını önceleyen alan etiketleri ve bağlam ipuçları (v3)
_PII_NAME_LABELS = (r"(?:Ad[ıi]?\s*[-/]?\s*Soyad[ıi]?|Ad\s+Soyad|M[üu]şteri(?:\s+Ad[ıi])?|Hesap\s+Sahibi|Kart\s+Sahibi|Al[ıi]c[ıi]|G[öo]nderen|"
                    r"Kefil|Borçlu|Vekil|Yetkili|[İI]lgili\s+Ki[şs]i|Talep\s+Sahibi|Ba[şs]vuru\s+Sahibi|Ad[ıi])")
_PII_HONORIFIC = r"(?:Sayın|Sn\.|Bay|Bayan|Sayin)"
_PII_SEP = r"[\s.\-]?"

def _pii_digits(s):
    return re.sub(r"\D", "", s)

def _pii_tc_ok(d):
    if len(d) != 11 or d[0] == "0": return False
    n = list(map(int, d))
    return n[9] == ((sum(n[0:9:2]) * 7) - sum(n[1:8:2])) % 10 and n[10] == sum(n[:10]) % 10

def _pii_luhn_ok(d):
    if not 13 <= len(d) <= 19: return False
    s = 0
    for i, ch in enumerate(reversed(d)):
        x = int(ch)
        if i % 2 == 1:
            x *= 2
            if x > 9: x -= 9
        s += x
    return s % 10 == 0

def _pii_iban_ok(raw):
    s = re.sub(r"[\s\-]", "", raw).upper()
    if not re.fullmatch(r"TR\d{24}", s): return False
    r = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1

def _filter_pii(text: str, mappings_dict: dict = None) -> str:
    """Kişisel veri maskeleme v2: ayraç toleranslı desenler + sağlama doğrulaması + ad sözlüğü."""
    original_text = text
    global_mappings = {} if mappings_dict is None else mappings_dict
    spans = []  # (start, end, category)

    def add(m, cat, group=0):
        s, e = m.span(group)
        if not any(s < e2 and s2 < e for s2, e2, _ in spans):
            spans.append((s, e, cat))

    # 1) IBAN (TR + 24 hane, boşluk/tire toleranslı, büyük-küçük harf duyarsız, mod-97 doğrulamalı)
    for m in re.finditer(r"(?i)\bTR[\s\-]?\d{2}(?:[\s\-]?\d){22}\b", text):
        if _pii_iban_ok(m.group()): add(m, "IBAN")
    # 2) E-posta
    for m in re.finditer(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", text):
        add(m, "EMAIL")
    # 3) Kart numarası: gerçek kart düzeni (4-4-4-4 veya 4-6-5, tutarlı ayraç), Luhn doğrulamalı
    for m in re.finditer(r"(?<!\d)(?<!\d[.,])(?:\d{4}([\s.\-]?)\d{4}\1\d{4}\1\d{4}|\d{4}([\s\-]?)\d{6}\2\d{5})(?!\d|[.,]\d)", text):
        d = _pii_digits(m.group())
        if _pii_luhn_ok(d) and not _pii_tc_ok(d): add(m, "CREDIT_CARD")
    # 4) TC kimlik no (11 hane, ayraç toleranslı, sağlama doğrulamalı)
    for m in re.finditer(r"(?<!\d)(?<!\d[.,])[1-9](?:[\s.\-]?\d){10}(?!\d|[.,]\d)", text):
        if _pii_tc_ok(_pii_digits(m.group())): add(m, "TC_NO")
    # 5) Cep telefonu (+90 / 0 önekli veya öneksiz 5xx; boşluk, nokta, tire, parantez toleranslı)
    for m in re.finditer(r"(?<![\d])(?:\+?90[\s.\-]?)?(?:0[\s.\-]?)?\(?5\d{2}\)?[\s.\-]?\d{3}[\s.\-]?\d{2}[\s.\-]?\d{2}(?![\d])", text):
        d = _pii_digits(m.group())
        if len(d) in (10, 11, 12) and d[-10] == "5": add(m, "PHONE")
    # 6) Ad-soyad: (a) unvan + 1-3 büyük harfli sözcük, (b) sözlükteki bir ad + büyük harfli/tamamı büyük soyad,
    #    (c) TAMAMI BÜYÜK soyad + sözlükteki ad
    W = r"[A-ZÇĞİÖŞÜ][a-zçğıöşü]+"; U = r"[A-ZÇĞİÖŞÜ]{2,}"
    for m in re.finditer(rf"\b{_PII_HONORIFIC}\s+((?:{W}|{U})(?:\s+(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    # (d) alan etiketi + ":" + 1-3 büyük harfli sözcük (aynı satır); Kefil/Borçlu/Vekil için ":" isteğe bağlı  [v3]
    SP = r"[ \t]+"
    for m in re.finditer(rf"(?i:\b{_PII_NAME_LABELS})[ \t]*[:：][ \t]*((?:{W}|{U})(?:{SP}(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    for m in re.finditer(rf"\b(?:Kefil|Borçlu|Vekil|Sayın|Sn\.)[ \t]+((?:{W}|{U})(?:{SP}(?:{W}|{U})){{0,2}})", text):
        add(m, "NAME", 1)
    # (e) 2-3 büyük harfli sözcük (aynı satır) + isimli/adlı/adındaki  [v3]
    for m in re.finditer(rf"\b((?:{W}|{U})(?:{SP}(?:{W}|{U})){{1,2}})(?:'\w+)?[ \t]+(?:isimli|adlı|adındaki)\b", text):
        add(m, "NAME", 1)
    # (f) sözlükteki ad + TAMAMI BÜYÜK soyad (aynı satır; Türkçe belge geleneği)  [v3]
    for m in re.finditer(rf"\b({W})[ \t]+({U})\b", text):
        if _pii_lower(m.group(1)) in _PII_FIRST_NAMES and m.group(2) not in _PII_NOT_SURNAME_UP: add(m, "NAME")
    # (b) sözlükteki ad + 0-1 ara ad + soyad: her büyük harfli sözcük konumundan ayrı ayrı denenir (örtüşen eşleşme)
    for m in re.finditer(rf"(?=\b({W})((?:\s+{W})?)\s+({W}|{U})\b)", text):
        first, mid, last = m.group(1), m.group(2), m.group(3)
        if _pii_lower(first) in _PII_FIRST_NAMES and last not in _PII_NOT_SURNAME and mid.strip() not in _PII_NOT_SURNAME:
            s0 = m.start(1); e0 = m.end(3); spans_add = (s0, e0)
            if not any(s0 < e2 and s2 < e0 for s2, e2, _ in spans): spans.append((s0, e0, "NAME"))
    # (c) TAMAMI BÜYÜK soyad + sözlükteki ad
    for m in re.finditer(rf"\b({U})\s+({W})\b", text):
        if _pii_lower(m.group(2)) in _PII_FIRST_NAMES: add(m, "NAME")

    if not spans:
        return text

    def placeholder(val, cat):
        for p, orig in global_mappings.items():
            if orig == val: return p
        n = sum(1 for p in global_mappings if p.startswith(f"[{cat}_")) + 1
        p = f"[{cat}_{n}]"; global_mappings[p] = val
        return p

    out = []; last = 0
    for s, e, cat in sorted(spans):
        out.append(text[last:s]); out.append(placeholder(text[s:e], cat)); last = e
    out.append(text[last:])
    text = "".join(out)
    if os.getenv("ENABLE_PII_LOGGING", "false").lower() == "true":
        logger.info(f"[PII Masking] Original: '{original_text}'")
    logger.info(f"[PII Masking] Masking complete. Items changed: {len(spans)}")
    return text
