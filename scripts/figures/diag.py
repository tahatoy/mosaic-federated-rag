"""Architecture diagrams (Figures 1-3) redrawn for the final configuration, EN + TR, same visual style as the originals."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.environ.get("MOSAIC_FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "Liberation Serif", "savefig.dpi": 330})
LW = 2.0
G1, G2 = "#f0f0f0", "#dcdcdc"


class D:
    def __init__(self, w, h, figw):
        self.fig = plt.figure(figsize=(figw, figw * h / w))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")

    def container(self, x, y, w, h, title, size=13):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18", fc=G1, ec="black", lw=LW, ls=(0, (4, 2.2))))
        self.ax.text(x + 0.22, y + h - 0.2, title, ha="left", va="top", fontsize=size, fontweight="bold")

    def box(self, x, y, w, h, title, lines=(), fill="white", tsize=12, lsize=10.5, gap=0.44):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec="black", lw=LW))
        n = len(lines)
        first = y + h / 2 + gap * n / 2
        self.ax.text(x + w / 2, first, title, ha="center", va="center", fontsize=tsize, fontweight="bold")
        for i, ln in enumerate(lines):
            self.ax.text(x + w / 2, first - gap * (i + 1), ln, ha="center", va="center", fontsize=lsize)

    def arrow(self, pts, both=False, dashed=False, head=0.16):
        xs, ys = zip(*pts)
        ls = (0, (4, 2.2)) if dashed else "-"
        self.ax.plot(xs[:-1] + (xs[-1],), ys, color="black", lw=LW, ls=ls, solid_capstyle="butt")
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color="black", lw=LW, mutation_scale=16, shrinkA=0, shrinkB=0))
        if both:
            self.ax.annotate("", xy=pts[0], xytext=pts[1], arrowprops=dict(arrowstyle="-|>", color="black", lw=LW, mutation_scale=16, shrinkA=0, shrinkB=0))

    def label(self, x, y, s, size=11.5, bg=None, ha="center"):
        self.ax.text(x, y, s, ha=ha, va="center", fontsize=size, style="italic",
                     bbox=dict(fc=bg, ec="none", pad=2) if bg else None)

    def save(self, name):
        for ext in ("png", "pdf"):
            self.fig.savefig(f"{OUT}/{name}.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.05)
        plt.close(self.fig)


L = {
    "en": dict(
        hq="Headquarters (HQ): RTX 5060 Ti 16 GB, Ryzen 5 7500F, 32 GB RAM", ui=("Web UI", ["React (8005)"]),
        orch=("Orchestrator", ["FastAPI, SSE (8004)", "PII masking"]), router=("BERT Router", ["ONNX INT8 (8000)"]),
        shared=("Shared RAG (8003)", ["ChromaDB, 429 chunks"]), rr=("Reranker", ["cross-encoder", "(mMARCO)"]),
        qwen=("Qwen3-8B Q4_K_M", ["llama-server (8098)"]), gemma=("Gemma-3-4B Q4_K_M", ["llama-server (8099)"]),
        fl=("Flower Server", ["gRPC/TLS (9080)", "FedAvg / FedProx", "μ = 0.01"]),
        path="Query path: UI → orchestrator → router → federated retrieval → re-ranking → LLM",
        rest="REST/mutual TLS: query and anonymized text chunks",
        b1="Branch-1 VM (Retail): 2 vCPU, 10 GB RAM", b2="Branch-2 VM (Corporate): 2 vCPU, 10 GB RAM",
        db1=("Local ChromaDB", ["7,854 chunks"]), db2=("Local ChromaDB", ["14,767 chunks"]),
        api1=("Branch-1 RAG API (8001)", ["MOSAIC-Embed,", "local PII filter"]), api2=("Branch-2 RAG API (8002)", ["MOSAIC-Embed,", "local PII filter"]),
        flc=("Flower Client", ["local FL training"]), bui=("Isolated Branch UI", ["React (8005)"]),
        grpc="gRPC/TLS: model/adapter updates and global weights",
        # fig 2
        hq2="Headquarters (HQ)", tb="Target Branch (Branch-1 VM)",
        s1=("1. User Query", ['e.g., "What is the mortgage', 'early repayment penalty?"']),
        s2=("2. Routing", ["BERT label: complex", "query sent to all branches"]),
        s3=("3. Cross-Branch Request", ["async HTTPX, mutual TLS", "branch APIs (8001, 8002)"]),
        s4=("4. Local Search", ["MOSAIC-Embed + ChromaDB", "top 8 candidates per service"]),
        s5=("5. Late Masking", ["TCKN, name, IBAN masked", "only anonymous text returned"]),
        s6=("6. Re-ranking", ["candidates from all branches", "up to 6 chunks (score ≥ 0.20)"]),
        s7=("7. Prompt & Inference", ["context + masked query", "Qwen3-8B (llama-server)"]),
        s8=("8. Streamed Answer (SSE)", ["masks restored locally", "answer with cited sources"]),
        req="request (mutual TLS)", anon="anonymized chunks", raw="Raw PDFs and personal data\nnever leave the branch VM.",
        # fig 3
        uq="User Query", bc=("BERT Classifier", ["dbmdz/bert-base-turkish-cased, ONNX INT8 (25.6 ms)"]), lc="label + confidence",
        rl=("Rule-Based Safeguard Layer", ["Rule 1: if any of 18 complex analysis keywords appears (e.g., dcf, duration, swap, stress test) → complex",
                                         "Rule 2: if prediction confidence < 0.55, promote one tier (chitchat → simple, simple → complex)"]),
        cc=("chitchat", ["Gemma-3-4B", "no retrieval"]), sm=("simple", ["Federated RAG + re-ranking", "+ Gemma-3-4B"]),
        cx=("complex", ["Federated RAG + re-ranking", "+ Qwen3-8B"]),
        foot="To reduce error cost, uncertain cases are routed to the stronger model.",
    ),
    "tr": dict(
        hq="Genel merkez (HQ): RTX 5060 Ti 16 GB, Ryzen 5 7500F, 32 GB RAM", ui=("Arayüz", ["React (8005)"]),
        orch=("Orkestratör", ["FastAPI, SSE (8004)", "PII maskeleme"]), router=("BERT yönlendirici", ["ONNX INT8 (8000)"]),
        shared=("Ortak RAG (8003)", ["ChromaDB, 429 ortak parça"]), rr=("Yeniden sıralayıcı", ["çapraz kodlayıcı", "(mMARCO)"]),
        qwen=("Qwen3-8B Q4_K_M", ["llama-server (8098)"]), gemma=("Gemma-3-4B Q4_K_M", ["llama-server (8099)"]),
        fl=("Flower sunucusu", ["gRPC/TLS (9080)", "FedAvg / FedProx", "μ = 0.01"]),
        path="Sorgu yolu: arayüz → orkestratör → yönlendirici → federe bilgi getirme → yeniden sıralama → dil modeli",
        rest="REST/karşılıklı TLS: maskelenmiş sorgu ve anonim metin parçaları",
        b1="Şube-1 VM (bireysel): 2 vCPU, 10 GB RAM", b2="Şube-2 VM (kurumsal): 2 vCPU, 10 GB RAM",
        db1=("Yerel ChromaDB", ["7.854 parça"]), db2=("Yerel ChromaDB", ["14.767 parça"]),
        api1=("Şube-1 RAG API (8001)", ["MOSAIC-Embed,", "yerel PII filtresi"]), api2=("Şube-2 RAG API (8002)", ["MOSAIC-Embed,", "yerel PII filtresi"]),
        flc=("Flower istemcisi", ["yerel LoRA eğitimi"]), bui=("İzole şube arayüzü", ["React (8005)"]),
        grpc="gRPC/TLS: adaptör güncellemeleri ve global model ağırlıkları",
        hq2="Genel merkez (HQ)", tb="Hedef şube (Şube-1 VM)",
        s1=("1. Kullanıcı sorgusu", ['Örn. "Konut kredisi erken', 'ödeme tazminatı nedir?"']),
        s2=("2. Yönlendirme", ["BERT etiketi: complex", "sorgu tüm şubelere gider"]),
        s3=("3. Çapraz şube isteği", ["asenkron HTTPX, karşılıklı TLS", "şube API'leri (8001, 8002)"]),
        s4=("4. Yerel arama", ["MOSAIC-Embed + ChromaDB", "servis başına ilk 8 aday"]),
        s5=("5. Geç maskeleme", ["TCKN, ad, IBAN maskelenir", "yalnızca anonim parça döner"]),
        s6=("6. Yeniden sıralama", ["tüm şubelerin adayları", "en fazla 6 parça (skor ≥ 0.20)"]),
        s7=("7. İstem ve çıkarım", ["bağlam + maskeli sorgu", "Qwen3-8B (llama-server)"]),
        s8=("8. Akışlı yanıt (SSE)", ["maskeler yerelde çözülür", "kaynak gösterimli yanıt"]),
        req="istek (karşılıklı TLS)", anon="anonim parçalar", raw="Ham PDF ve kişisel veriler\nşube sanal makinesinden çıkmaz.",
        uq="Kullanıcı sorgusu", bc=("BERT sınıflandırıcı", ["dbmdz/bert-base-turkish-cased, ONNX INT8 (25.6 ms)"]), lc="etiket + güven",
        rl=("Kural tabanlı koruma katmanı", ["Kural 1: 18 karmaşık analiz anahtar kelimesinden biri varsa (ör. dcf, durasyon, swap, stres testi) → complex",
                                              "Kural 2: tahmin güveni < 0.55 ise bir üst sınıfa terfi (chitchat → simple, simple → complex)"]),
        cc=("chitchat", ["Gemma-3-4B", "bilgi getirme yok"]), sm=("simple", ["Şube RAG (MOSAIC-Embed)", "+ Gemma-3-4B"]),
        cx=("complex", ["Federe RAG + yeniden sıralama", "+ Qwen3-8B"]),
        foot="Hata maliyetini azaltmak için belirsiz durumda rota daha güçlü modele yükseltilir.",
    ),
}


def fig1(lang):
    t = L[lang]; d = D(20, 10.9, 10.0)
    d.container(0.2, 5.75, 19.15, 5.0, t["hq"])
    d.box(0.45, 6.75, 2.45, 1.3, *t["ui"], fill=G2)
    d.box(3.5, 6.6, 3.2, 1.6, *t["orch"], fill=G2)
    d.box(3.5, 8.85, 3.2, 1.3, *t["router"])
    d.box(7.3, 8.85, 3.7, 1.3, *t["shared"])
    d.box(7.3, 6.6, 3.7, 1.6, *t["rr"])
    d.box(11.6, 8.85, 3.7, 1.3, *t["qwen"])
    d.box(11.6, 6.6, 3.7, 1.6, *t["gemma"])
    d.box(15.9, 6.6, 3.05, 3.55, *t["fl"], fill=G2, gap=0.46)
    d.arrow([(2.9, 7.4), (3.5, 7.4)], both=True); d.arrow([(6.7, 7.4), (7.3, 7.4)], both=True)
    d.arrow([(5.1, 8.2), (5.1, 8.85)], both=True); d.arrow([(9.15, 8.2), (9.15, 8.85)], both=True)
    d.arrow([(11.0, 7.4), (11.6, 7.4)]); d.arrow([(11.3, 7.4), (11.3, 9.5), (11.6, 9.5)])
    d.label(10.2, 6.18, t["path"], size=11.5, bg=G1)
    d.container(0.2, 0.55, 9.3, 4.55, t["b1"]); d.container(10.05, 0.55, 9.3, 4.55, t["b2"])
    for x0, db, api in ((0.2, t["db1"], t["api1"]), (10.05, t["db2"], t["api2"])):
        d.box(x0 + 0.25, 2.85, 4.15, 1.4, *db); d.box(x0 + 4.85, 2.85, 4.2, 1.4, *api)
        d.box(x0 + 0.25, 0.9, 4.15, 1.4, *t["flc"], fill=G2); d.box(x0 + 4.85, 0.9, 4.2, 1.4, *t["bui"])
        d.arrow([(x0 + 4.4, 3.55), (x0 + 4.85, 3.55)], both=True); d.arrow([(x0 + 6.95, 2.3), (x0 + 6.95, 2.85)], both=True)
    d.arrow([(5.1, 6.6), (5.1, 5.35), (8.2, 5.35), (8.2, 4.25)])
    d.arrow([(8.2, 5.35), (18.15, 5.35), (18.15, 4.25)])
    d.label(11.6, 5.35, t["rest"], bg="white")
    d.arrow([(2.3, 0.15), (2.3, 0.9)], dashed=True); d.arrow([(12.1, 0.15), (12.1, 0.9)], dashed=True)
    d.ax.plot([2.3, 19.65, 19.65], [0.15, 0.15, 8.35], color="black", lw=LW, ls=(0, (4, 2.2)))
    d.arrow([(19.65, 8.35), (18.95, 8.35)], dashed=True)
    d.label(10.8, 0.15, t["grpc"], bg="white")
    d.save(f"fig01_architecture_{lang}")


def fig2(lang):
    t = L[lang]; d = D(20, 9.9, 9.0)
    d.container(0.25, 4.05, 19.5, 5.6, t["hq2"]); d.container(0.25, 0.25, 19.5, 3.3, t["tb"])
    d.box(0.5, 6.85, 4.3, 1.7, *t["s1"]); d.box(5.35, 6.85, 4.3, 1.7, *t["s2"]); d.box(10.2, 6.85, 4.3, 1.7, *t["s3"], lsize=10)
    d.box(0.5, 4.5, 4.3, 1.7, *t["s8"], fill=G2, tsize=11.5); d.box(5.35, 4.5, 4.3, 1.7, *t["s7"], fill=G2); d.box(10.2, 4.5, 4.3, 1.7, *t["s6"], lsize=10)
    d.box(10.2, 0.75, 4.3, 1.7, *t["s5"]); d.box(15.05, 0.75, 4.3, 1.7, *t["s4"])
    d.arrow([(4.8, 7.7), (5.35, 7.7)]); d.arrow([(9.65, 7.7), (10.2, 7.7)])
    d.arrow([(14.5, 7.7), (17.2, 7.7), (17.2, 2.45)]); d.label(17.2, 5.2, t["req"], bg=G1)
    d.arrow([(15.05, 1.6), (14.5, 1.6)]); d.arrow([(12.35, 2.45), (12.35, 4.5)]); d.label(12.35, 3.8, t["anon"], bg=G1)
    d.arrow([(10.2, 5.35), (9.65, 5.35)]); d.arrow([(5.35, 5.35), (4.8, 5.35)])
    d.ax.text(5.1, 1.75, t["raw"], ha="center", va="center", fontsize=13, style="italic", linespacing=1.4)
    d.save(f"fig02_workflow_{lang}")


def fig3(lang):
    t = L[lang]; d = D(20, 9.9, 9.0)
    d.box(7.3, 8.8, 5.4, 1.0, t["uq"], fill=G2, tsize=13)
    d.box(4.95, 6.95, 10.1, 1.3, *t["bc"])
    d.box(0.55, 4.6, 18.9, 1.8, *t["rl"], fill=G1, lsize=10.6, gap=0.5)
    d.box(0.55, 1.2, 5.8, 2.05, *t["cc"]); d.box(7.1, 1.2, 5.8, 2.05, *t["sm"]); d.box(13.65, 1.2, 5.8, 2.05, *t["cx"], fill=G2)
    d.arrow([(10, 8.8), (10, 8.25)]); d.arrow([(10, 6.95), (10, 6.4)]); d.label(10.15, 6.68, t["lc"], ha="left")
    d.arrow([(7.4, 4.6), (3.45, 3.25)]); d.arrow([(10, 4.6), (10, 3.25)]); d.arrow([(12.6, 4.6), (16.55, 3.25)])
    d.label(10, 0.55, t["foot"], size=13)
    d.save(f"fig03_router_logic_{lang}")


if __name__ == "__main__":
    for lang in ("en", "tr"):
        fig1(lang); fig2(lang); fig3(lang)
    print("ok")
