"""
MOSAIC Finansal Danışman Arayüzü
=================================
"Türkçe Finans İçin Federe RAG Tabanlı Düşük Kaynaklı Adaptif Chatbot"

Özellikler:
  - Şube kimliği seçimi (Bireysel / Kurumsal)
  - Router karar rozeti + güven skoru
  - Federe gizlilik göstergesi (hangi ChromaDB kullanıldı)
  - RAG kaynak paneli (kaynak adı + skor)
  - Latency ölçümü
  - Demo senaryoları (tek tık)
  - Çapraz şube erişim simülasyonu

Port: 8005
"""

import os
import json
import time
import httpx
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path

# .env dosyasını yükle (root dizinden)
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8004")
BRANCH1_RAG_URL  = os.getenv("BRANCH1_RAG_URL",  "http://127.0.0.1:8001")
BRANCH2_RAG_URL  = os.getenv("BRANCH2_RAG_URL",  "http://127.0.0.1:8002")
HQ_RAG_URL       = os.getenv("HQ_RAG_URL",        "http://127.0.0.1:8003")

# ─── SAYFA YAPISI ─────────────────────────────────────────────────────────────

def get_score_label(score: float) -> str:
    try:
        val = float(score)
    except Exception:
        return "Eşleşme"
    if val >= 0.55:
        return f"Güçlü Eşleşme (%{val*100:.0f})"
    elif val >= 0.40:
        return f"Orta Derece Eşleşme (%{val*100:.0f})"
    else:
        return f"Destekleyici Eşleşme (%{val*100:.0f})"


st.set_page_config(
    page_title="MOSAIC Finansal Danışman",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Kurumsal lacivert palet */
  [data-testid="stAppViewContainer"] {
    background: #070E1C;
  }
  [data-testid="stSidebar"] {
    background: #0D1A30;
    border-right: 1px solid #1E2F50;
  }
  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stRadio,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] small {
    color: #B0BDD0 !important;
  }
  [data-testid="stSidebar"] h3,
  [data-testid="stSidebar"] strong,
  [data-testid="stSidebar"] b {
    color: #E8EDF5 !important;
  }

  /* Genel metin rengi */
  .stMarkdown, p, label { color: #C8D3E0; }
  h1, h2, h3 { color: #E8EDF5; }

  /* Rozet stilleri */
  .badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 3px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 2px;
  }
  .badge-chitchat { background: #0E2A1A; color: #4DB87A; border: 1px solid #2A6645; }
  .badge-simple   { background: #2A2200; color: #C9A84C; border: 1px solid #6E5A1A; }
  .badge-complex  { background: #1A1630; color: #8B7FCC; border: 1px solid #4A3F88; }

  /* Federe gizlilik rozeti */
  .badge-private  { background: #0A1E34; color: #5B9BD5; border: 1px solid #1E4A7A; }
  .badge-global   { background: #0E2218; color: #4DB87A; border: 1px solid #1A5530; }
  .badge-blocked  { background: #2A1010; color: #CC6060; border: 1px solid #7A2020; }

  /* Kaynak kartı */
  .src-card {
    background: #0D1A30;
    border-left: 3px solid #1E4A7A;
    padding: 10px 14px;
    margin: 5px 0;
    border-radius: 3px;
    font-size: 0.80rem;
    line-height: 1.6;
    color: #B0BDD0;
  }
  .src-card b { color: #5B9BD5; }

  /* Federasyon topoloji kutusu */
  .topo-box {
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-radius: 4px;
    padding: 12px 10px;
    font-family: 'Courier New', monospace;
    font-size: 0.73rem;
    color: #7A8BA8;
    line-height: 1.9;
  }
  .topo-active { color: #4DB87A; font-weight: bold; }
  .topo-global { color: #C9A84C; font-weight: bold; }
  .topo-branch { color: #5B9BD5; font-weight: bold; }

  /* Metadata satırı */
  .meta-row {
    font-size: 0.72rem;
    color: #7A8BA8;
    margin-top: 8px;
    padding: 6px 10px;
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-radius: 3px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .meta-sep { color: #1E2F50; }

  /* Pipeline kutusu */
  .pipeline-box {
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-radius: 4px;
    padding: 12px 14px;
    margin-bottom: 10px;
  }
  .pipeline-label {
    font-size: 0.63rem;
    color: #4A5A70;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 8px;
    font-weight: 700;
  }

  /* Durum noktaları */
  .dot-wait    { color: #3A4A5A; }
  .dot-running { color: #C9A84C; }
  .dot-done    { color: #4DB87A; }
  .dot-error   { color: #CC6060; }
  .dot-skip    { color: #3A4A5A; }

  /* Karşılama kartları */
  .info-card {
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-top: 3px solid #1E4A7A;
    border-radius: 4px;
    padding: 16px 18px;
    color: #B0BDD0;
  }
  .info-card .stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #C9A84C;
    display: block;
    margin: 6px 0 2px;
  }
  .info-card .stat-lbl {
    font-size: 0.70rem;
    color: #4A5A70;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .info-card .card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #5B9BD5;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
  }

  /* Şube bilgi kartı */
  .branch-card {
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-radius: 4px;
    padding: 10px 12px;
    margin: 6px 0;
  }

  /* Sube baslik */
  .sidebar-header {
    font-size: 0.65rem;
    font-weight: 700;
    color: #4A5A70;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin: 12px 0 4px;
  }

  /* Chat mesaj arka plan */
  [data-testid="stChatMessage"] {
    background: #0D1A30 !important;
    border: 1px solid #1E2F50;
    border-radius: 4px;
    margin-bottom: 8px;
  }

  /* Markdown Tablo Stilleri (Satır ve Sütun Çizgileri) */
  table {
    width: 100% !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border: 1px solid #1E2F50 !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    margin: 14px 0 !important;
    background: #091426 !important;
  }
  th {
    background: #0D1A30 !important;
    color: #4DB87A !important;
    font-weight: 700 !important;
    padding: 10px 14px !important;
    border-bottom: 2px solid #1E2F50 !important;
    border-right: 1px solid #1E2F50 !important;
    text-align: left !important;
  }
  th:last-child { border-right: none !important; }
  td {
    padding: 9px 14px !important;
    border-bottom: 1px solid #1E2F50 !important;
    border-right: 1px solid #1E2F50 !important;
    color: #C8D3E0 !important;
  }
  td:last-child { border-right: none !important; }
  tbody tr:last-child td { border-bottom: none !important; }
  tbody tr:nth-child(even) td { background: rgba(255, 255, 255, 0.02) !important; }
  tbody tr:hover td { background: rgba(255, 255, 255, 0.05) !important; }

  /* Stremlit butonlar */
  .stButton button {
    background: #0D1A30;
    color: #B0BDD0;
    border: 1px solid #1E2F50;
    border-radius: 3px;
    font-size: 0.75rem;
    transition: border-color 0.15s, color 0.15s;
  }
  .stButton button:hover {
    border-color: #C9A84C;
    color: #C9A84C;
  }

  /* Divider */
  hr { border-color: #1E2F50 !important; }

  /* Expander */
  [data-testid="stExpander"] {
    background: #0D1A30;
    border: 1px solid #1E2F50;
    border-radius: 4px;
  }

  /* Caption/small */
  .stCaption { color: #4A5A70 !important; }

  /* File uploader */
  [data-testid="stFileUploader"] {
    background: #0D1A30;
    border: 1px dashed #1E2F50;
    border-radius: 4px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #070E1C; }
  ::-webkit-scrollbar-thumb { background: #1E2F50; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────────

BRANCH_INFO = {
    1: {
        "name": "Şube-1",
        "title": "Bireysel Bankacılık",
        "label": "BİR",
        "color": "#1E4A7A",
        "color_text": "#5B9BD5",
        "db": "finance_branch1",
        "rag_port": 8001,
        "rag_url": BRANCH1_RAG_URL,
        "desc": "Kredi, mevduat, bireysel emeklilik, kart ürünleri",
        "doc_total": 628,
        "doc_unique": 537,
        "doc_overlap": 91,
        "topics": {
            "Bireysel / Kredi": {"count": 499, "color": "#1E4A7A", "color_text": "#5B9BD5", "keywords": ["konut kredisi", "taşıt kredisi", "tüketici kredisi", "kredi kartı", "mevduat", "vadesiz hesap", "BES", "bireysel emeklilik", "sigorta", "IBAN"]},
            "Kurumsal / Yatırım": {"count": 129, "color": "#2A3A50", "color_text": "#7A8BA8", "keywords": ["repo", "tahvil", "portföy", "fon"]},
        },
    },
    2: {
        "name": "Şube-2",
        "title": "Kurumsal Bankacılık",
        "label": "KUR",
        "color": "#1A4A2A",
        "color_text": "#4DB87A",
        "db": "finance_branch2",
        "rag_port": 8002,
        "rag_url": BRANCH2_RAG_URL,
        "desc": "İşletme kredisi, FX risk, IPO, kurumsal yatırım",
        "doc_total": 983,
        "doc_unique": 892,
        "doc_overlap": 91,
        "topics": {
            "Kurumsal / Yatırım": {"count": 688, "color": "#1A4A2A", "color_text": "#4DB87A", "keywords": ["döviz kuru", "FX risk", "repo", "swap", "opsiyon", "tahvil", "IPO", "halka arz", "kurumsal finansman", "işletme kredisi"]},
            "Bireysel / Kredi": {"count": 295, "color": "#2A3A50", "color_text": "#7A8BA8", "keywords": ["kredi", "mevduat", "faiz"]},
        },
    },
}

binfo = BRANCH_INFO[1]  # Safe default to prevent NameError

GLOBAL_DB = {
    "name": "finance_global",
    "desc": "FedProx ile eğitilmiş paylaşılmış bilgi tabanı (1611 belge)",
}

ROUTE_META = {
    "chitchat": {
        "badge": "badge-chitchat",
        "label": "GENEL SOHBET",
        "db_used": "YOK — RAG cagirilmadi",
        "privacy": "badge-global",
        "privacy_label": "FEDERASYONDAN BAGIMSIZ",
    },
    "simple": {
        "badge": "badge-simple",
        "label": "ŞUBE SORGUSU",
        "db_used": None,
        "privacy": "badge-private",
        "privacy_label": "ŞUBEYE ÖZGÜ VERİ",
    },
    "complex": {
        "badge": "badge-complex",
        "label": "DERİN ANALİZ",
        "db_used": GLOBAL_DB["name"],
        "privacy": "badge-global",
        "privacy_label": "FEDERE GLOBAL VERI",
    },
}

DEMO_SCENARIOS = {
    "Selamlasma": {
        "query": "Merhaba! Finans konusunda yardim alabilir miyim?",
        "branch": 0,
        "aciklama": "Router chitchat tanır — RAG yok, hızlı yanıt",
    },
    "Bireysel Kredi (Şube-1)": {
        "query": "Konut kredisi faiz oranı nasıl hesaplanır?",
        "branch": 1,
        "aciklama": "Şube-1 (Bireysel) özel ChromaDB kullanılır",
    },
    "Şube 2'nin Operasyonel Süreçleri": {
        "query": "Şube 2'nin operasyonel süreçlerinden sorumlu baş denetçi kimdir?",
        "branch": 0,
        "aciklama": "TEST: Sorgu metninde 'Şube 2' var → otomatik şube tespiti",
    },
    "Kurumsal Kredi (Şube-2)": {
        "query": "Kurumsal kredi başvurusunda hangi belgeler gerekir?",
        "branch": 2,
        "aciklama": "Şube-2 (Kurumsal) özel ChromaDB kullanılır",
    },
    "Şube 1'den Bireysel Ürünler": {
        "query": "Şube 1'de sunulan bireysel emeklilik ürünleri nelerdir?",
        "branch": 0,
        "aciklama": "TEST: Sorgu metninde 'Şube 1' var → otomatik şube tespiti",
    },
    "Global Analiz": {
        "query": "TCMB para politikası kararlarının döviz kuru ve kredi büyümesine etkisini analiz edin.",
        "branch": 0,
        "aciklama": "Complex soru — global federe veri her şubeye eşit erişim",
    },
    "Portföy Çeşitlendirmesi": {
        "query": "Enflasyon ortamında portföy çeşitlendirmesi ve reel getiri optimizasyonu stratejileri nelerdir?",
        "branch": 0,
        "aciklama": "Makro + çok bileşenli — complex — global ChromaDB",
    },
    "Döviz Riski Yönetimi": {
        "query": "Kurumsal bir şirket için döviz riski yönetimi ve hedge stratejileri nasıl kurgulanır?",
        "branch": 0,
        "aciklama": "Kurumsal risk analizi — complex — global federe veri",
    },
    "IPO Süreci": {
        "query": "Halka arz (IPO) sürecinde yüklenici bankalar ve fiyat belirleme nasıl işler?",
        "branch": 0,
        "aciklama": "Yatırım bankacılığı — complex — global bilgi tabanı",
    },
}

# ─── MÜŞTERİ PROFİLLERİ ───────────────────────────────────────────────────────

MUSTERI_PROFILLERI = {
    "Ayşe H. — Genç Aile": {
        "profil": "Evli, 2 çocuk  |  Öğretmen (net 28.000 TL)  |  İlk eve sahip olma planı",
        "branch": 1,
        "sorgular": [
            ("Konut kredisi için minimum ne kadar birikimim olmalı?", "simple"),
            ("28.000 TL maaşla 2.5 milyon TL konut kredisi çekebilir miyim?", "simple"),
            ("Konut kredisi alırken DASK sigortası zorunlu mu?", "simple"),
        ],
        "tez_notu": "Şube-1 özel verisi — bireysel kredi knowhow'u gizli tutulur",
    },
    "Kaya Ltd. — KOBİ Üretici": {
        "profil": "Tekstil üreticisi  |  45 çalışan  |  EUR cinsinden ihracat gelirleri",
        "branch": 2,
        "sorgular": [
            ("EUR/TRY kur riskini forward kontratla nasıl hedge edebiliriz?", "complex"),
            ("KOBİ ihracat teşvik kredisi başvurusu için gerekli belgeler nelerdir?", "simple"),
            ("4 milyon TRY işletme kredisi için teminat olarak stok gösterilebilir mi?", "simple"),
        ],
        "tez_notu": "FX/hedge — complex Qwen3-8B  |  kredi belgesi — simple Gemma-3-4B",
    },
    "Mehmet B. — Emeklilik Planlaması": {
        "profil": "55 yaşında  |  8 yıl sonra emekliliği planlıyor  |  Mevcut birikim 600.000 TL",
        "branch": 1,
        "sorgular": [
            ("BES'e şimdi girsem 8 yılda devlet katkısıyla ne kadar birikim olur?", "simple"),
            ("Enflasyona karşı koruyucu tahvil ve altın sepeti portföyü nasıl kurulur?", "complex"),
        ],
        "tez_notu": "BES — Şube-1 verisi  |  Portföy analizi — global federe bilgi",
    },
    "TechStart A.Ş. — Girişim": {
        "profil": "SaaS startup  |  2 yıllık  |  Seed round tamamlandı  |  Büyüme finansmanı arıyor",
        "branch": 2,
        "sorgular": [
            ("Girişim şirketi için halka arz öncesi (pre-IPO) süreç nasıl işler?", "complex"),
            ("Venture debt ile equity round arasındaki vergisel farklar nelerdir?", "complex"),
        ],
        "tez_notu": "Tamamen complex — Qwen3-8B + global ChromaDB (1611 belge)",
    },
    "Çiftçi Kooperatifi — Tarım": {
        "profil": "120 üye çiftçi  |  Tarımsal üretim  |  Mevsimsel nakit akışı sorunu",
        "branch": 1,
        "sorgular": [
            ("Tarımsal destek kredisi faiz sübvansiyonu nedir?", "simple"),
            ("Hasat dönemi öncesi döner kredi limiti nasıl açılır?", "simple"),
        ],
        "tez_notu": "Bireysel kredi tipi sorular — Şube-1 finance_branch1 DB",
    },
}

PRIVACY_MATRIX = {
    (1, "simple"):   ("badge-private", "Şube-1 Özel Verisi",  "[ÖZEL]"),
    (2, "simple"):   ("badge-private", "Şube-2 Özel Verisi",  "[ÖZEL]"),
    (1, "complex"):  ("badge-global",  "Federe Global Veri",  "[GLOBAL]"),
    (2, "complex"):  ("badge-global",  "Federe Global Veri",  "[GLOBAL]"),
    (1, "chitchat"): ("badge-chitchat","RAG Yok",             "[DİREKT]"),
    (2, "chitchat"): ("badge-chitchat","RAG Yok",             "[DİREKT]"),
}


def query_orchestrator_stream(query: str, branch_id: int, history: list[dict] | None = None):
    """SSE stream'i okur, her adimi yield eder. Son olay 'result' step'idir."""
    payload: dict = {"query": query, "branch_id": branch_id}
    if history:
        payload["history"] = history
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{ORCHESTRATOR_URL}/query/stream",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    yield json.loads(line)


def query_orchestrator(query: str, branch_id: int) -> dict:
    """Eski uyumluluk icin — /query (blok) endpoint'ini cagirир."""
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            f"{ORCHESTRATOR_URL}/query",
            json={"query": query, "branch_id": branch_id},
        )
        r.raise_for_status()
        return r.json()


def health_check() -> dict:
    services = {}
    with httpx.Client(timeout=4.0) as client:
        for name, port in [("router", 8000), ("rag", 8003), ("orchestrator", 8004), ("llm", 8099)]:
            try:
                r = client.get(f"http://localhost:{port}/health")
                services[name] = "ok" if r.status_code == 200 else f"http:{r.status_code}"
            except httpx.ConnectError:
                services[name] = "down"
            except Exception as e:
                services[name] = "error"
    return services


def render_message(msg: dict):
    """Sohbet geçmişinden bir mesajı render et."""
    role     = msg["role"]
    content  = msg["content"]
    meta     = msg.get("meta", {})
    branch   = msg.get("branch", 1)

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and meta:
            route      = meta.get("route", "simple")
            conf       = meta.get("confidence", 0)
            elapsed    = meta.get("elapsed_ms", 0)
            sources    = meta.get("sources", [])
            model_name = meta.get("model_name", "Qwen3-8B" if route == "complex" else "Gemma-3-4B")
            rm         = ROUTE_META.get(route, ROUTE_META["simple"])
            priv_cls, priv_lbl, priv_icon = PRIVACY_MATRIX.get(
                (branch, route), ("badge-global", "?", "?")
            )

            if route == "simple":
                db_name = BRANCH_INFO[branch]["db"]
            elif route == "complex":
                db_name = GLOBAL_DB["name"]
            else:
                db_name = "—"

            llm_cls = "badge-complex" if route == "complex" else "badge-simple"
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="badge {rm["badge"]}">{rm["label"]}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span class="badge {priv_cls}">{priv_icon} {priv_lbl}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span class="badge {llm_cls}">{model_name}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span style="color:#5B9BD5;font-family:monospace">{db_name}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span>{elapsed:.0f} ms</span>'
                f'<span class="meta-sep">|</span>'
                f'<span>Güven: {conf:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander(f"Kaynaklar — {len(sources)} belge", expanded=False):
                    for i, src in enumerate(sources, 1):
                        txt  = src.get("content", "")[:280]
                        name = src.get("source", "bilinmiyor")
                        score= src.get("score", 0)
                        label = get_score_label(score)
                        st.markdown(
                            f'<div class="src-card">'
                            f'<b>[{i}] {name}</b> &nbsp;&nbsp; {label}<br>'
                            f'{txt}{"..." if len(src.get("content",""))>280 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ─── SESSION STATE ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "pending_branch" not in st.session_state:
    st.session_state.pending_branch = 1


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="padding:18px 4px 14px;">'
        '<div style="font-size:1.05rem;font-weight:700;color:#E8EDF5;letter-spacing:0.04em">MOSAIC</div>'
        '<div style="font-size:0.68rem;color:#4A5A70;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px">Finansal Danışman Sistemi</div>'
        '<div style="font-size:0.63rem;color:#2A3A50;margin-top:4px;font-family:monospace">Federe RAG  |  Türkçe Finans  |  v1.0</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Şube seçimi
    st.markdown('<div class="sidebar-header">Sorgu Kapsamı</div>', unsafe_allow_html=True)
    branch_id = st.radio(
        "Şube",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "HQ — Tüm Şubeler (Global Arama)",
            1: f"{BRANCH_INFO[1]['name']} — {BRANCH_INFO[1]['title']}",
            2: f"{BRANCH_INFO[2]['name']} — {BRANCH_INFO[2]['title']}"
        }[x],
        label_visibility="collapsed",
        help="0: Tüm şubelerde arama yap | 1,2: Şube-spesifik arama + HQ kaynakları"
    )

    # Set binfo for UI elements that need branch info (default to Branch 1 for global mode)
    binfo = BRANCH_INFO[branch_id if branch_id != 0 else 1]

    if branch_id == 0:
        st.markdown(
            '<div class="branch-card" style="border-left:3px solid #5B5B5B;">'
            '<b style="color:#9BA9B8;font-size:0.82rem">HQ — Global Arama</b><br>'
            '<span style="color:#7A8BA8;font-size:0.72rem">Tüm şubelerin verileri paralel olarak taranır (KVKK: Ham veri şubelerde kalır)</span><br>'
            '<span style="font-family:monospace;color:#5B9BD5;font-size:0.70rem">finance_global + tüm branch db\'ler</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="branch-card" style="border-left:3px solid {binfo["color"]};">'
            f'<b style="color:{binfo["color_text"]};font-size:0.82rem">{binfo["name"]}</b>'
            f'<span style="color:#4A5A70;font-size:0.68rem;margin-left:8px">{binfo["title"]}</span><br>'
            f'<span style="color:#7A8BA8;font-size:0.72rem">{binfo["desc"]}</span><br>'
            f'<span style="font-family:monospace;color:#5B9BD5;font-size:0.70rem">{binfo["db"]}</span>'
            f'<span style="color:#4A5A70;font-size:0.68rem;margin-left:8px">{binfo["doc_total"]} belge</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Federe topoloji
    st.markdown('<div class="sidebar-header">Federe Mimari</div>', unsafe_allow_html=True)

    if branch_id == 0:
        # HQ Global view
        st.markdown(
            f'<div class="topo-box">'
            f'<span class="topo-global">HQ — Global (aktif)</span><br>'
            f'&nbsp; finance_global<br>'
            f'&nbsp;&nbsp; | Paralel Query<br>'
            f'<span class="topo-branch">Şube-1</span><br>'
            f'&nbsp; finance_branch1 ({BRANCH_INFO[1]["doc_total"]} doc)<br>'
            f'&nbsp;&nbsp; | Paralel Query<br>'
            f'<span class="topo-branch">Şube-2</span><br>'
            f'&nbsp; finance_branch2 ({BRANCH_INFO[2]["doc_total"]} doc)<br>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        other = 2 if branch_id == 1 else 1
        ob = BRANCH_INFO[other]
        st.markdown(
            f'<div class="topo-box">'
            f'<span class="topo-active">{binfo["name"]} (aktif)</span><br>'
            f'&nbsp; {binfo["db"]}<br>'
            f'&nbsp;&nbsp; | FedProx<br>'
            f'<span class="topo-global">finance_global (HQ)</span><br>'
            f'&nbsp;&nbsp; | FedProx<br>'
            f'{ob["name"]}<br>'
            f'&nbsp; {ob["db"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption("Şubeler arası özel veri transferi yoktur (KVKK/GDPR uyumlu).")

    st.divider()

    # Şube veri haritası
    st.markdown('<div class="sidebar-header">Şube Veri Haritası</div>', unsafe_allow_html=True)

    with st.expander(f"{binfo['name']} İçeriği", expanded=True):
        for topic_name, tdata in binfo["topics"].items():
            bar_pct = int(tdata["count"] / binfo["doc_total"] * 100)
            st.markdown(
                f'<div style="margin:4px 0;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.70rem;">'
                f'<span style="color:{tdata["color_text"]};font-weight:600">{topic_name}</span>'
                f'<span style="color:#4A5A70">{tdata["count"]} ({bar_pct}%)</span>'
                f'</div>'
                f'<div style="background:#0A1628;border-radius:2px;height:4px;margin:2px 0 5px;">'
                f'<div style="background:{tdata["color_text"]};opacity:0.6;width:{bar_pct}%;height:4px;border-radius:2px;"></div>'
                f'</div>'
                f'<div style="font-size:0.63rem;color:#3A4A5A;line-height:1.7;">'
                + ' '.join(f'<code style="background:#0A1628;padding:1px 4px;border-radius:2px;color:#5A7A9A">{kw}</code>' for kw in tdata["keywords"][:6])
                + '</div></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Global DB (HQ)", expanded=False):
        st.markdown(
            '<div style="font-size:0.72rem;color:#7A8BA8;">'
            '<b style="color:#C9A84C">1520 belge</b> (Şube-1 + Şube-2 birleşik)<br>'
            '<span style="color:#4A5A70">537</span> Ş-1 özgün &nbsp;·&nbsp; '
            '<span style="color:#4A5A70">892</span> Ş-2 özgün &nbsp;·&nbsp; '
            '<span style="color:#4A5A70">91</span> ortak<br>'
            'Tüm konular: bireysel, kurumsal, yatırım, FX, kredi, BES'
            '</div>',
            unsafe_allow_html=True,
        )

    # Get current branch info for display (works after sidebar definitions)
    _current_binfo = BRANCH_INFO[branch_id if branch_id != 0 else 1]

    st.markdown(
        f'<div style="background:#0A1628;border:1px solid #1E2F50;border-radius:3px;padding:7px 10px;margin:6px 0;font-size:0.70rem;color:#7A8BA8;">'
        f'<span class="badge badge-simple" style="font-size:0.60rem">ŞUBE SORGUSU</span> '
        f'<span style="font-family:monospace;color:#5B9BD5">{_current_binfo["db"]}</span><br>'
        f'<span class="badge badge-complex" style="font-size:0.60rem">DERİN ANALİZ</span> '
        f'<span style="font-family:monospace;color:#C9A84C">finance_global</span><br>'
        f'<span class="badge badge-chitchat" style="font-size:0.60rem">GENEL SOHBET</span> '
        f'RAG yok, doğrudan LLM'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Demo senaryolar
    st.markdown('<div class="sidebar-header">Demo Senaryoları</div>', unsafe_allow_html=True)
    for label, scen in DEMO_SCENARIOS.items():
        if st.button(label, use_container_width=True, key=f"demo_{label}"):
            st.session_state.pending_query  = scen["query"]
            st.session_state.pending_branch = scen["branch"]
            st.rerun()

    st.divider()

    # Müşteri portföy senaryoları
    st.markdown('<div class="sidebar-header">Müşteri Portföy Senaryoları</div>', unsafe_allow_html=True)
    st.caption("Gerçek hayat müşteri profillerinden sorgular")
    for profil_label, pdata in MUSTERI_PROFILLERI.items():
        with st.expander(profil_label, expanded=False):
            st.markdown(
                f'<div style="font-size:0.70rem;color:#7A8BA8;margin-bottom:6px;line-height:1.6;">'
                f'{pdata["profil"]}</div>',
                unsafe_allow_html=True,
            )
            st.caption(pdata.get("tez_notu", ""))
            for i, (q_text, q_type) in enumerate(pdata["sorgular"]):
                lbl_cls = "badge-complex" if q_type == "complex" else ("badge-chitchat" if q_type == "chitchat" else "badge-simple")
                btn_key = f"musteri_{profil_label}_{i}"
                col_btn, col_badge = st.columns([3, 1])
                with col_btn:
                    if st.button(f"{q_text[:55]}..." if len(q_text) > 55 else q_text, key=btn_key, use_container_width=True):
                        st.session_state.pending_query  = q_text
                        st.session_state.pending_branch = pdata["branch"]
                        st.rerun()
                with col_badge:
                    st.markdown(
                        f'<div style="padding-top:6px"><span class="badge {lbl_cls}" style="font-size:0.58rem">{q_type}</span></div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # Federe belge yükleme
    st.markdown('<div class="sidebar-header">Federe Belge Yükleme</div>', unsafe_allow_html=True)
    st.caption(
        "Şubeye özel belge yükle. Ham veri yalnızca bu şubenin ChromaDB'sine gider. "
        "Federe senkronizasyon ile bilgi global'e de aktarılır — ama ham veri değil."
    )

    with st.expander(f"{binfo['name']} — Belge Ekle", expanded=False):
        uploaded_file = st.file_uploader(
            "PDF veya TXT seç",
            type=["pdf", "txt"],
            key=f"upload_file_{branch_id}",
        )
        src_name = st.text_input(
            "Kaynak adı (opsiyonel)",
            placeholder="örn: konut_kredi_kilavuzu",
            key=f"src_name_{branch_id}",
        )

        if uploaded_file is not None:
            st.markdown(
                f'<div style="background:#0A1628;border:1px solid #1E2F50;border-radius:3px;'
                f'padding:7px 10px;font-size:0.72rem;color:#7A8BA8;margin:4px 0;">'
                f'<b style="color:#C9A84C">{uploaded_file.name}</b> '
                f'({uploaded_file.size // 1024} KB)'
                f'</div>',
                unsafe_allow_html=True,
            )

        col_priv, col_fed = st.columns(2)
        with col_priv:
            btn_private = st.button(
                f"[ÖZEL] {binfo['name']}",
                key=f"upload_private_{branch_id}",
                use_container_width=True,
                disabled=uploaded_file is None,
                help=f"Yalnızca {binfo['db']} koleksiyonuna ekler.",
            )
        with col_fed:
            btn_federate = st.button(
                "[GLOBAL] Federe Senkronize",
                key=f"upload_federate_{branch_id}",
                use_container_width=True,
                disabled=uploaded_file is None,
                help="Şubeye ekler + FedProx agregasyonu simüle eder.",
            )

        def _do_upload(to_global: bool):
            branch_url  = binfo["rag_url"]
            global_url  = HQ_RAG_URL
            src         = src_name.strip() or uploaded_file.name.rsplit(".", 1)[0]
            file_bytes  = uploaded_file.getvalue()
            file_tuple  = (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")

            results = {}
            with httpx.Client(timeout=60.0) as c:
                try:
                    r1 = c.post(
                        f"{branch_url}/upload",
                        files={"file": file_tuple},
                        data={"source_name": src},
                    )
                    r1.raise_for_status()
                    results["branch"] = r1.json()
                except httpx.ConnectError as e:
                    results["branch_error"] = (
                        f"RAG servisi bağlantı hatası ({branch_url}). "
                        f"Lütfen tüm servislerin çalıştığını kontrol et."
                    )
                except Exception as e:
                    err_msg = str(e)
                    if "getaddrinfo" in err_msg or "DNS" in err_msg:
                        results["branch_error"] = (
                            f"DNS çözümleme hatası ({branch_url}). "
                            f"Servislerin localhost üzerinde çalıştığını doğrula."
                        )
                    else:
                        results["branch_error"] = f"Yükleme hatası: {err_msg[:100]}"

                if to_global and "branch_error" not in results:
                    try:
                        r2 = c.post(
                            f"{global_url}/upload",
                            files={"file": (uploaded_file.name, file_bytes, file_tuple[2])},
                            data={"source_name": src},
                        )
                        r2.raise_for_status()
                        results["global"] = r2.json()
                    except httpx.ConnectError as e:
                        results["global_error"] = (
                            f"Federe RAG bağlantı hatası ({global_url}). "
                            f"Global servisi başlat."
                        )
                    except Exception as e:
                        err_msg = str(e)
                        if "getaddrinfo" in err_msg or "DNS" in err_msg:
                            results["global_error"] = (
                                f"DNS çözümleme hatası ({global_url}). "
                                f"Lütfen ortam değişkenlerini kontrol et."
                            )
                        else:
                            results["global_error"] = f"Federe yükleme hatası: {err_msg[:100]}"

            return results

        if btn_private or btn_federate:
            with st.spinner("Yukleniyor..."):
                res = _do_upload(to_global=btn_federate)

            if "branch_error" in res:
                st.error(f"Şube yükleme hatası: {res['branch_error']}")
            else:
                bd = res.get("branch", {})
                st.success(
                    f"{binfo['name']} — {bd.get('chunks_added', '?')} chunk eklendi "
                    f"(toplam: {bd.get('collection_total', '?')})"
                )

            if btn_federate:
                if "global_error" in res:
                    st.error(f"Global senkronizasyon hatası: {res['global_error']}")
                elif "global" in res:
                    gd = res.get("global", {})
                    st.success(
                        f"Global — {gd.get('chunks_added', '?')} chunk senkronize edildi "
                        f"(toplam: {gd.get('collection_total', '?')})"
                    )
                    st.info(
                        "Federe Gizlilik: Ham belge yalnızca şube ChromaDB'sinde. "
                        "Global'e yalnızca embedding/chunk kopyası gitti — orijinal veri değil.",
                    )

        st.markdown(
            '<div style="margin-top:8px;padding:8px 10px;background:#0A1628;border-left:3px solid #1E4A7A;'
            'border-radius:3px;font-size:0.68rem;color:#7A8BA8;line-height:1.8;">'
            '<b style="color:#5B9BD5">Tez Senaryosu</b><br>'
            '1. Şube-1\'e konut kredi kılavuzu yükle (sadece şube)<br>'
            '2. Şube-2\'den aynı konuyu sorgula — yanıt yok (gizlilik korunur)<br>'
            '3. Şube-1\'den "Federe Senkronize" et<br>'
            '4. Şube-2\'den complex sorgula — finance_global\'den yanıt gelir'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Sistem sağlığı
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if st.button("Sistem Durumu", use_container_width=True):
            svcs = health_check()
            for s, v in svcs.items():
                st.session_state[f"health_{s}"] = v
    with col_h2:
        if st.button("Geçmişi Temizle", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for s in ["router", "rag", "orchestrator", "llm"]:
        v = st.session_state.get(f"health_{s}", "?")
        dot_color = "#4DB87A" if v == "ok" else ("#CC6060" if v == "error" else "#3A4A5A")
        st.markdown(
            f'<span style="color:{dot_color};font-size:0.9rem">&#9679;</span>'
            f'<span style="color:#7A8BA8;font-size:0.72rem;margin-left:4px">{s}: {v}</span><br>',
            unsafe_allow_html=True,
        )


# ─── ANA İÇERİK ───────────────────────────────────────────────────────────────

binfo = BRANCH_INFO[branch_id if branch_id != 0 else 1]
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown(
        f'<div style="padding:10px 0 8px;">'
        f'<span style="font-size:1.3rem;font-weight:700;color:#E8EDF5;letter-spacing:0.02em">MOSAIC Finansal Danışman</span>'
        f'&nbsp;&nbsp;<span style="font-size:0.72rem;color:{binfo["color_text"]};background:#0A1628;'
        f'padding:3px 10px;border-radius:2px;border:1px solid {binfo["color"]};font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.06em">'
        f'{binfo["name"]} / {binfo["title"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with col_t2:
    st.markdown(
        '<div style="text-align:right;padding-top:14px;">'
        '<span style="color:#3A4A5A;font-size:0.68rem;font-family:monospace">'
        'Gemma-3-4B · Qwen3-8B · Q4_K_M · FedProx'
        '</span></div>',
        unsafe_allow_html=True,
    )

# Karşılama kartları (ilk açılışta)
if not st.session_state.messages:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">Yönlendirici Router — BERT</div>'
            '<div class="stat-lbl">Yönlendirici Model</div>'
            '<span class="stat-val">F1 = 0.8509</span>'
            '<div style="font-size:0.72rem;color:#7A8BA8;margin-top:4px">'
            'FedProx ile eğitilmiş 3-sınıf yönlendirici<br>'
            'chitchat / simple / complex'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">RAG Pipeline</div>'
            '<div class="stat-lbl">Bilgi Erişimi</div>'
            '<span class="stat-val">Hit@10 = 1.000</span>'
            '<div style="font-size:0.72rem;color:#7A8BA8;margin-top:4px">'
            'Fine-tuned multilingual-e5-small<br>'
            'ChromaDB + CrossEncoder reranker'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="info-card">'
            '<div class="card-title">Üretken Modeller</div>'
            '<div class="stat-lbl">Yanıt Kalitesi</div>'
            '<span class="stat-val">ROUGE-L = 0.182</span>'
            '<div style="font-size:0.72rem;color:#7A8BA8;margin-top:4px">'
            'Gemma-3-4B Q4 — simple/chitchat<br>'
            'Qwen3-8B Q4 — complex'
            '</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div style="margin:12px 0;padding:10px 14px;background:#0A1628;'
        'border:1px solid #1E2F50;border-radius:3px;font-size:0.78rem;color:#7A8BA8;">'
        '<b style="color:#C9A84C">Demo Kullanımı:</b> Sol panelden bir senaryo seçin '
        'veya doğrudan sorunuzu yazın. Her yanıta router kararı, '
        'kullanılan ChromaDB koleksiyonu ve federe gizlilik durumu eklenir.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

# Mesaj geçmişi
for msg in st.session_state.messages:
    render_message(msg)

# Pending demo sorgusu
if st.session_state.pending_query:
    incoming_query  = st.session_state.pending_query
    incoming_branch = st.session_state.pending_branch
    st.session_state.pending_query  = None
    st.session_state.pending_branch = 1
else:
    incoming_query  = None
    incoming_branch = branch_id


# ─── GİRİŞ ────────────────────────────────────────────────────────────────────

# Use branch_id to get current branch info for chat input placeholder
current_binfo = BRANCH_INFO[branch_id if branch_id != 0 else 1]
user_input = st.chat_input(
    f"{current_binfo['name']} — sorunuzu yazin",
    key="chat_input",
)

active_query  = incoming_query or user_input
active_branch = incoming_branch if incoming_query else branch_id

if active_query:
    # Kullanıcı mesajı
    st.session_state.messages.append({
        "role": "user",
        "content": active_query,
        "branch": active_branch,
    })
    with st.chat_message("user"):
        st.markdown(active_query)
        if incoming_query:
            branch_label = f"{BRANCH_INFO[active_branch]['name']}"
            st.caption(f"Demo: {branch_label} olarak sorgulanıyor")

    # Asistan yanıtı — canlı pipeline adımları
    with st.chat_message("assistant"):
        # Pipeline durum gösterimi
        pipeline_box = st.empty()
        answer_box   = st.empty()

        step_state = {"router": "wait", "rag": "wait", "llm": "wait"}
        data = {}

        def render_pipeline(extra_html=""):
            dot = {"wait": ("&#9675;", "#3A4A5A"), "running": ("&#9679;", "#C9A84C"),
                   "done": ("&#9679;", "#4DB87A"), "error": ("&#9679;", "#CC6060"),
                   "skip": ("&#9675;", "#3A4A5A")}
            labels = {
                "router": ("[ 1 ] BERT Router",   step_state.get("_router_detail", "3-sınıflı yönlendirici")),
                "rag":    ("[ 2 ] RAG / ChromaDB", step_state.get("_rag_detail",    "vektör arama")),
                "llm":    ("[ 3 ] LLM Üreteci",   step_state.get("_llm_detail",    "Gemma-3-4B")),
            }
            rows = ""
            for s, (title, detail) in labels.items():
                st8 = step_state.get(s, "wait")
                sym, col = dot[st8]
                rows += (
                    f'<div style="display:flex;align-items:baseline;gap:8px;padding:4px 0;'
                    f'border-bottom:1px solid #0A1628;">'
                    f'<span style="color:{col};font-size:0.85rem">{sym}</span>'
                    f'<span style="font-size:0.78rem;font-weight:700;color:{col};font-family:monospace;min-width:140px">{title}</span>'
                    f'<span style="font-size:0.70rem;color:#4A5A70">{detail}</span>'
                    f'</div>'
                )
            pipeline_box.markdown(
                f'<div class="pipeline-box">'
                f'<div class="pipeline-label">İŞLEME SÜRECİ</div>'
                f'{rows}{extra_html}</div>',
                unsafe_allow_html=True,
            )

        render_pipeline()
        accumulated_answer = ""

        # Konuşma geçmişini hazırla: mevcut soruyu hariç tut (henüz yanıtlanmadı)
        # Son 6 mesaj = 3 tur (orchestrator kendi max_turns kuralını ayrıca uygular)
        prior_messages = st.session_state.messages[:-1][-6:]
        history_payload = [
            {"role": m["role"], "content": m["content"]}
            for m in prior_messages
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        try:
            for event in query_orchestrator_stream(
                active_query, active_branch,
                history=history_payload if history_payload else None,
            ):
                step   = event.get("step")
                status = event.get("status")

                if step == "router":
                    if status == "running":
                        step_state["router"] = "running"
                        step_state["_router_detail"] = "analiz ediliyor..."
                        step_state["_router_t0"] = time.perf_counter()
                    elif status == "done":
                        route  = event["route"]
                        conf   = event["confidence"]
                        router_ms = event.get("elapsed_ms") or round((time.perf_counter() - step_state.get("_router_t0", time.perf_counter())) * 1000, 1)
                        step_state["_router_elapsed_ms"] = router_ms
                        step_state["router"] = "done"
                        step_state["_router_detail"] = f"{route.upper()}  |  güven {conf:.0%}  |  {router_ms:.0f} ms"
                        if event.get("label") == 0:
                            step_state["rag"] = "skip"
                            step_state["_rag_detail"] = "atlandı (chitchat)"
                    elif status == "error":
                        step_state["router"] = "error"

                elif step == "rag":
                    if status == "running":
                        step_state["rag"] = "running"
                        step_state["_rag_detail"] = event.get("msg", "sorgulanıyor...")
                        step_state["_rag_t0"] = time.perf_counter()
                    elif status == "done":
                        db_fetched = event.get("db", "")
                        doc_count  = event.get("doc_count", 0)
                        top_score  = event.get("top_score", 0)
                        # Saf RAG süresi: toplam elapsed - router elapsed
                        total_ms  = event.get("elapsed_ms", 0)
                        router_ms = step_state.get("_router_elapsed_ms", 0)
                        rag_ms    = (total_ms - router_ms) if total_ms > router_ms else round((time.perf_counter() - step_state.get("_rag_t0", time.perf_counter())) * 1000, 1)
                        step_state["_rag_elapsed_ms"] = total_ms
                        step_state["rag"] = "done"
                        label_detail = get_score_label(top_score)
                        step_state["_rag_detail"] = f"{db_fetched}  |  {doc_count} belge  |  {label_detail}  |  {rag_ms:.0f} ms"

                elif step == "llm":
                    if status == "running":
                        step_state["llm"] = "running"
                        step_state["_llm_detail"] = event.get("msg", "üretiyor...")
                        step_state["_llm_t0"] = time.perf_counter()
                    elif status == "done":
                        total_ms  = event.get("elapsed_ms", 0)
                        rag_ms    = step_state.get("_rag_elapsed_ms", 0)
                        llm_ms    = (total_ms - rag_ms) if total_ms > rag_ms else round((time.perf_counter() - step_state.get("_llm_t0", time.perf_counter())) * 1000, 1)
                        mdl = event.get("model_name", "Gemma-3-4B")
                        step_state["llm"] = "done"
                        step_state["_llm_detail"] = f"{mdl}  |  {llm_ms:.0f} ms"

                elif step == "llm_token":
                    accumulated_answer += event.get("token", "")
                    answer_box.markdown(accumulated_answer + "▌")
                    continue

                elif step == "result" and status == "done":
                    data = event
                    if accumulated_answer:
                        answer_box.markdown(accumulated_answer)

                render_pipeline()

        except Exception as exc:
            pipeline_box.error(f"Bağlantı hatası: {exc}")
            st.session_state.messages.append({"role": "assistant", "content": f"Hata: {exc}", "branch": active_branch, "meta": {}})
            st.rerun()

        if data:
            route      = data.get("route", "simple")
            answer     = data.get("answer", accumulated_answer) or accumulated_answer
            conf       = data.get("confidence", 0)
            elapsed    = data.get("elapsed_ms", 0)
            sources    = data.get("sources", [])
            model_name = data.get("model_name", "Gemma-3-4B")
            rm         = ROUTE_META.get(route, ROUTE_META["simple"])
            priv_cls, priv_lbl, priv_icon = PRIVACY_MATRIX.get(
                (active_branch, route), ("badge-global", "?", "?")
            )
            if route == "simple":
                db_name = BRANCH_INFO[active_branch]["db"]
            elif route == "complex":
                db_name = GLOBAL_DB["name"]
            else:
                db_name = "—"
            llm_cls = "badge-complex" if route == "complex" else "badge-simple"

            answer_box.markdown(answer)
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="badge {rm["badge"]}">{rm["label"]}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span class="badge {priv_cls}">{priv_icon} {priv_lbl}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span class="badge {llm_cls}">{model_name}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span style="color:#5B9BD5;font-family:monospace;font-size:0.72rem">{db_name}</span>'
                f'<span class="meta-sep">|</span>'
                f'<span>{elapsed:.0f} ms</span>'
                f'<span class="meta-sep">|</span>'
                f'<span>Güven: {conf:.0%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if sources:
                with st.expander(f"Kaynaklar — {len(sources)} belge", expanded=False):
                    for i, src in enumerate(sources, 1):
                        txt   = src.get("content", "")[:280]
                        name  = src.get("source", "bilinmiyor")
                        score = src.get("score", 0)
                        label = get_score_label(score)
                        st.markdown(
                            f'<div class="src-card">'
                            f'<b>[{i}] {name}</b> &nbsp;&nbsp; {label}<br>'
                            f'{txt}{"..." if len(src.get("content",""))>280 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "branch":  active_branch,
                "meta": {
                    "route":      route,
                    "confidence": conf,
                    "elapsed_ms": elapsed,
                    "sources":    sources,
                    "model_name": model_name,
                },
            })
