"""MOSAIC Q1 figures (EN + TR). Data: ai/experiments/results/q1/*.json (values cross-checked with the tables)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

R = os.environ.get("MOSAIC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ai", "experiments", "results", "q1", ""))
OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, GRAY_D = "#a9a8a3", "#6b6a66"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
W = 6.3  # inch, full text width

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False, "hatch.linewidth": 0.6,
})


def J(name):
    return json.load(open(R + name))


def save(fig, name, lang):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}_{lang}.{ext}", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def panel(ax, letter):
    ax.text(-0.01, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=8.5)


T = {
    "en": dict(
        acc="Accuracy (%)", tmpl="Template test set (n = 3,000)", ood="Out-of-template questions (expected class: simple)",
        maj="Majority class", tfidf="TF-IDF + logistic regression", e5lr="e5-small + logistic regression",
        cen="BERTurk, centralized (pooled)", fa="BERTurk, FedAvg", fp="BERTurk, FedProx", loc="BERTurk, local (either branch)",
        dep="Deployed router, ONNX INT8", share="Share of questions (%)",
        bt="BERT router, templated (n = 600)", bp="BERT router, paraphrased (n = 419)",
        tt="TF-IDF, templated (n = 600)", tp="TF-IDF, paraphrased (n = 419)",
        cx="complex", si="simple", ch="chitchat",
        mrr="MRR@10 (federated setting, 95% CI)", templ="Templated (n = 600)", para="Paraphrased (n = 419)",
        mem="GPU memory", speed="Generation speed", em_q="Cross-branch EM (same contexts)", gib="GiB", tps="Tokens/s",
        file="Model file", vram="GPU memory", em="EM (%)", nt="not\ntested",
        h1a="Templated questions (n = 500)", h1b="Paraphrased questions (n = 352)", h1c="Live end-to-end (n = 200, paraphrased)",
        emci="Exact match (%, 95% CI)",
        A="No retrieval", B="Asking branch only", C1="MOSAIC, initial model", D1="Central index, initial model", C3="MOSAIC, MOSAIC-Embed", D3="Central index, MOSAIC-Embed",
        C2="MOSAIC, re-indexed (base e5-small)", D2="Central index, base e5-small",
        L0="Qwen3-8B without retrieval", L1="MOSAIC, initial model", L4="MOSAIC, MOSAIC-Embed", L2="MOSAIC, re-indexed (run 1)", L3="MOSAIC, re-indexed (run 2)",
        g_base="Baselines", g_fed="MOSAIC (federated)", g_cen="Central index (upper bound)",
        eps="ε over three rounds (δ = 10⁻⁵, log scale)", util="MRR@10 (mean ± SD, 3 seeds)",
        cdp="Central DP (noise added once at HQ)", ldp="Local DP (noise added by each client)",
        nodp="No DP", base="Base e5-small (untrained)",
        zt="Templated questions (n = 600)", zp="Paraphrased questions (n = 419)",
        raw="Raw text", root="Root reduction", morph="Morpheme segmentation", zmrr="MRR@10 (95% CI)",
        tek="Initial fine-tuned e5-small", rrf="BM25 + initial fine-tuned (RRF)", tekce="Initial fine-tuned + CE",
    ),
    "tr": dict(
        acc="Doğruluk (%)", tmpl="Kalıp test kümesi (n = 3.000)", ood="Kalıp dışı sorular (beklenen sınıf: basit)",
        maj="Çoğunluk sınıfı", tfidf="TF-IDF + lojistik regresyon", e5lr="e5-small + lojistik regresyon",
        cen="BERTurk, merkezi (birleşik veri)", fa="BERTurk, FedAvg", fp="BERTurk, FedProx", loc="BERTurk, yerel (her iki şube)",
        dep="Konuşlandırılan yönlendirici, ONNX INT8", share="Soruların payı (%)",
        bt="BERT yönlendirici, kalıp (n = 600)", bp="BERT yönlendirici, yeniden ifade (n = 419)",
        tt="TF-IDF, kalıp (n = 600)", tp="TF-IDF, yeniden ifade (n = 419)",
        cx="karmaşık", si="basit", ch="gündelik",
        mrr="MRR@10 (federe düzen, %95 GA)", templ="Kalıp sorular (n = 600)", para="Yeniden ifade (n = 419)",
        mem="Grafik bellek", speed="Üretim hızı", em_q="Çapraz şube EM (aynı bağlam)", gib="GiB", tps="Token/s",
        file="Model dosyası", vram="Grafik bellek", em="EM (%)", nt="sınanmadı",
        h1a="Kalıp sorular (n = 500)", h1b="Yeniden ifade edilmiş sorular (n = 352)", h1c="Canlı uçtan uca (n = 200, yeniden ifade)",
        emci="Tam eşleşme (%, %95 GA)",
        A="Bilgi getirme yok", B="Yalnız soran şube", C1="MOSAIC, ilk model", D1="Merkezi dizin, ilk model", C3="MOSAIC, MOSAIC-Embed", D3="Merkezi dizin, MOSAIC-Embed",
        C2="MOSAIC, yeniden indekslenmiş (taban e5-small)", D2="Merkezi dizin, taban e5-small",
        L0="Bilgi getirmesiz Qwen3-8B", L1="MOSAIC, ilk model", L4="MOSAIC, MOSAIC-Embed", L2="MOSAIC, yeniden indekslenmiş (1. koşu)", L3="MOSAIC, yeniden indekslenmiş (2. koşu)",
        g_base="Temel koşullar", g_fed="MOSAIC (federe)", g_cen="Merkezi dizin (üst sınır)",
        eps="Üç turda ε (δ = 10⁻⁵, logaritmik ölçek)", util="MRR@10 (ortalama ± SS, 3 tohum)",
        cdp="Merkezi DP (gürültü HQ'da bir kez)", ldp="Yerel DP (gürültü her istemcide)",
        nodp="DP yok", base="Taban e5-small (eğitimsiz)",
        zt="Kalıp sorular (n = 600)", zp="Yeniden ifade edilmiş sorular (n = 419)",
        raw="Ham metin", root="Kök indirgeme", morph="Morfem bölütleme", zmrr="MRR@10 (%95 GA)",
        tek="İlk ince ayarlı e5-small", rrf="BM25 + ilk ince ayarlı (RRF)", tekce="İlk ince ayarlı + CE",
    ),
}


# ---------------------------------------------------------------- Figure 4: router
def fig_router(lang):
    t = T[lang]; e4 = J("e4_router.json"); e9 = J("e9_router_ood.json")
    s = e4["federated"]["summary"]; b = e4["baselines"]
    rows = [(t["maj"], b["majority"]["acc"], GRAY), (t["tfidf"], b["tfidf_logreg"]["acc"], GRAY),
            (t["e5lr"], b["e5small_logreg"]["acc"], GRAY), (t["cen"], s["central_pooled"]["acc"][0], ORANGE),
            (t["fa"], s["FedAvg"]["acc"][0], BLUE), (t["fp"], s["FedProx"]["acc"][0], BLUE),
            (t["loc"], s["local_b1"]["acc"][0], GRAY_D), (t["dep"], b["bert_router_deployed"]["acc"], AQUA)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 4.3), gridspec_kw=dict(height_ratios=[8, 4.6], hspace=0.62))
    y = range(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        a1.barh(yi, 100 * v, height=0.62, color=c, edgecolor="white", lw=0.8)
        a1.text(100 * v + 1.5, yi, f"{100 * v:.1f}", va="center", fontsize=7, color=INK)
    a1.set_yticks(list(y)); a1.set_yticklabels([r[0] for r in rows]); a1.set_xlim(0, 112); a1.set_xticks([0, 25, 50, 75, 100])
    a1.set_xlabel(t["acc"]); a1.set_title(t["tmpl"], loc="left"); grid_x(a1); panel(a1, "a")
    sets = [(t["bt"], e9["sets"]["cloze_question"]["bert_with_rules"]), (t["bp"], e9["sets"]["paraphrased"]["bert_with_rules"]),
            (t["tt"], e9["sets"]["cloze_question"]["tfidf_logreg"]), (t["tp"], e9["sets"]["paraphrased"]["tfidf_logreg"])]
    y2 = range(len(sets))[::-1]
    for yi, (lab, d) in zip(y2, sets):
        left = 0
        for k, c, h in (("simple", BLUE, ""), ("complex", ORANGE, "////"), ("chitchat", AQUA, "")):
            v = 100 * d.get(k, 0)
            if v <= 0: continue
            a2.barh(yi, v, left=left, height=0.6, color=c, edgecolor="white", lw=1.0, hatch=h)
            if v >= 8:
                a2.text(left + v / 2, yi, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if k != "chitchat" else INK,
                        bbox=dict(boxstyle="round,pad=0.12", fc=c, ec="none") if h else None)
            left += v
    a2.set_yticks(list(y2)); a2.set_yticklabels([s_[0] for s_ in sets]); a2.set_xlim(0, 100)
    a2.set_xlabel(t["share"]); a2.set_title(t["ood"], loc="left"); panel(a2, "b")
    a2.legend(handles=[Patch(fc=BLUE, label=t["si"]), Patch(fc=ORANGE, hatch="////", ec="white", label=t["cx"]), Patch(fc=AQUA, label=t["ch"])],
              ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.42), handlelength=1.4, columnspacing=1.2)
    save(fig, "fig04_router", lang)


# ---------------------------------------------------------------- Figure 5: retrieval benchmark
def fig_retrieval(lang):
    t = T[lang]; e1 = dict(J("e1_retrieval.json")["methods"])
    e12 = J("e12_eval_turkembed_v3.json")["methods"]
    e1["mosaic-embed"], e1["mosaic-embed+ce"] = e12["turkembed-v3"], e12["turkembed-v3+ce"]
    names = {"bm25": "BM25", "e5-small": "e5-small", "bge-m3": "BGE-M3 (dense)" if lang == "en" else "BGE-M3 (yoğun)",
             "turkembed-ft": t["tek"], "turkembed4retrieval": "TurkEmbed4Retrieval",
             "e5-small+ce": "e5-small + CE", "turkembed-ft+ce": t["tekce"],
             "mosaic-embed": "MOSAIC-Embed", "mosaic-embed+ce": "MOSAIC-Embed + CE"}
    keys = sorted(names, key=lambda k: e1[k]["natural"]["federated"]["mrr@10"][0])
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for i, k in enumerate(keys):
        for q, c, mk, off in (("cloze", BLUE, "o", 0.14), ("natural", ORANGE, "s", -0.14)):
            m, (lo, hi) = e1[k][q]["federated"]["mrr@10"]
            ax.plot([lo, hi], [i + off] * 2, color=c, lw=1.4, solid_capstyle="round")
            ax.plot(m, i + off, marker=mk, ms=5.5, color=c, mec="white", mew=0.8, ls="none")
        m = e1[k]["natural"]["federated"]["mrr@10"][0]
        ax.text(1.003, i - 0.14, f"{m:.3f}", va="center", fontsize=6.8, color=INK2)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([names[k] for k in keys])
    for lab in ax.get_yticklabels():
        if "CE" in lab.get_text(): lab.set_fontweight("bold")
    ax.set_xlim(0.6, 1.03); ax.set_ylim(-0.6, len(keys) - 0.4); ax.set_xlabel(t["mrr"]); grid_x(ax)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", mec="white", label=t["templ"]),
                       Line2D([], [], color=ORANGE, marker="s", mec="white", label=t["para"])],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.text(1.003, len(keys) - 0.35, t["para"].split(" (")[0], fontsize=6.5, color=INK2, va="bottom")
    save(fig, "fig05_retrieval", lang)


# ---------------------------------------------------------------- Figure 6: quantization
def fig_quant(lang):
    t = T[lang]; Q = J("e2_generation_quant.json")["sonuclar"]["cloze"]
    fm = ["Q4_0", "Q4_K_M", "Q2_K"]
    file_ = [2.20, 2.32, 1.61]; vram = [4.40, 4.52, 3.83]; tps = [53.4, 48.8, 53.6]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw=dict(wspace=0.45))
    a = axs[0]; x = range(3)
    a.bar([i - 0.19 for i in x], file_, 0.36, color=GRAY, edgecolor="white", lw=1, hatch="////", label=t["file"])
    a.bar([i + 0.19 for i in x], vram, 0.36, color=BLUE, edgecolor="white", lw=1, label=t["vram"])
    for i in x:
        a.text(i - 0.19, file_[i] + 0.08, f"{file_[i]:.2f}", ha="center", fontsize=6.5)
        a.text(i + 0.19, vram[i] + 0.08, f"{vram[i]:.2f}", ha="center", fontsize=6.5)
    a.set_xticks(list(x)); a.set_xticklabels(fm); a.set_ylabel(t["gib"]); a.set_ylim(0, 5.6); a.set_title(t["mem"], loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=6.6, handlelength=1.2, columnspacing=0.8); grid_y(a); panel(a, "a")
    a = axs[1]
    a.bar(list(x), tps, 0.55, color=BLUE, edgecolor="white")
    for i in x: a.text(i, tps[i] + 1, f"{tps[i]:.1f}", ha="center", fontsize=6.8)
    a.set_xticks(list(x)); a.set_xticklabels(fm); a.set_ylabel(t["tps"]); a.set_ylim(0, 65); a.set_title(t["speed"], loc="left"); grid_y(a); panel(a, "b")
    a = axs[2]
    for i, key in ((1, "C_base_q4km"), (2, "C_base_q2k")):
        m, (lo, hi) = Q[key]["em"]
        a.bar(i, 100 * m, 0.55, color=BLUE if i == 1 else ORANGE, edgecolor="white", hatch="" if i == 1 else "////")
        a.errorbar(i, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], color=INK, lw=0.9, capsize=2.5)
        a.text(i + 0.31, 100 * m, f"{100 * m:.1f}", va="center", fontsize=6.8)
    a.text(0, 4, t["nt"], ha="center", va="bottom", fontsize=6.5, color=INK2, style="italic")
    a.set_xticks(list(x)); a.set_xticklabels(fm); a.set_ylabel(t["em"]); a.set_ylim(0, 100); a.set_title(t["em_q"], loc="left"); grid_y(a); panel(a, "c")
    save(fig, "fig06_quant", lang)


# ---------------------------------------------------------------- Figure 7: cross-branch EM (H1)
def fig_h1(lang):
    t = T[lang]
    E = J("e2_generation.json")["sonuclar"]          # hardened configuration, initial model
    M = J("e2_generation_me.json")["sonuclar"]       # final configuration, MOSAIC-Embed (live services)
    V = J("e2_generation_v2.json")["sonuclar"]       # arm A (no retrieval) of the same 500/352 questions
    live1 = J("e5_e2e_judge.json")["summary"]; live3 = J("e5_e2e_judge_v3.json")["summary"]
    panels = [
        (t["h1a"], [(t["A"], V["cloze"]["A"]["em"], GRAY, ""), (t["B"], E["cloze"]["B"]["em"], GRAY, ""),
                    (t["C1"], E["cloze"]["C"]["em"], BLUE, ""), (t["D1"], E["cloze"]["D"]["em"], ORANGE, "////"),
                    (t["C3"], M["cloze"]["C"]["em"], BLUE, ""), (t["D3"], M["cloze"]["D_me"]["em"], ORANGE, "////")]),
        (t["h1b"], [(t["A"], V["natural"]["A"]["em"], GRAY, ""), (t["C1"], E["natural"]["C"]["em"], BLUE, ""),
                    (t["D1"], E["natural"]["D"]["em"], ORANGE, "////"), (t["C3"], M["natural"]["C"]["em"], BLUE, ""),
                    (t["D3"], M["natural"]["D_me"]["em"], ORANGE, "////")]),
        (t["h1c"], [(t["L0"], live3["norag_em"], GRAY, ""), (t["L1"], live1["em"], BLUE, ""), (t["L4"], live3["em"], BLUE, "")]),
    ]
    heights = [len(p[1]) for p in panels]
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.6), sharex=True, gridspec_kw=dict(height_ratios=heights, hspace=0.55))
    for ax, (title, rows), letter in zip(axs, panels, "abc"):
        y = list(range(len(rows)))[::-1]
        for yi, (lab, (m, ci), c, h) in zip(y, rows):
            lo, hi = ci
            ax.barh(yi, 100 * m, height=0.62, color=c, edgecolor="white", lw=0.8, hatch=h)
            ax.errorbar(100 * m, yi, xerr=[[100 * (m - lo)], [100 * (hi - m)]], color=INK, lw=0.9, capsize=2.2)
            ax.text(100 * hi + 1.2, yi, f"{100 * m:.1f}", va="center", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_title(title, loc="left"); grid_x(ax); panel(ax, letter)
        ax.set_xlim(0, 100)
    axs[-1].set_xlabel(t["emci"])
    axs[0].legend(handles=[Patch(fc=GRAY, label=t["g_base"]), Patch(fc=BLUE, label=t["g_fed"]), Patch(fc=ORANGE, hatch="////", ec="white", label=t["g_cen"])],
                  ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.12), handlelength=1.4)
    save(fig, "fig07_crossbranch", lang)


# ---------------------------------------------------------------- Figure 8: privacy-utility
def fig_dp(lang):
    t = T[lang]; d = J("e3_fl_embedding_qp.json")  # query–passage pairs (Supplementary Table S3)
    import statistics as st
    g = {}
    for r in d["runs"]:
        g.setdefault(r["name"] + (f"|{r['z']}" if "z" in r else ""), []).append(r)
    zs = sorted({r["z"] for r in d["runs"] if "z" in r})
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), sharey=False, gridspec_kw=dict(wspace=0.28))
    for ax, q, title, letter in ((axs[0], "cloze_mrr@10", t["zt"], "a"), (axs[1], "natural_mrr@10", t["zp"], "b")):
        for mode, c, mk, lab in (("central", BLUE, "o", t["cdp"]), ("local", ORANGE, "s", t["ldp"])):
            xs, ms, sd = [], [], []
            for z in zs:
                rows = g[f"dp_{mode}_K10_lora|{z}"]
                xs.append(rows[0]["epsilon"]); v = [r[q] for r in rows]; ms.append(st.mean(v)); sd.append(st.pstdev(v))
            ax.errorbar(xs, ms, yerr=sd, color=c, marker=mk, ms=5, mec="white", mew=0.8, lw=1.6, capsize=2, label=lab)
        nod = st.mean(r[q] for r in g["fedavg_K10_lora"]); base = g["base_e5_small"][0][q]
        ax.axhline(nod, color=INK2, lw=1.0, ls="--"); ax.axhline(base, color=GRAY, lw=1.0, ls=":")
        lo_, hi_ = sorted([(nod, t["nodp"]), (base, t["base"])])
        ax.text(2.1, lo_[0] - 0.02, lo_[1], fontsize=6.3, color=INK2, va="top")
        ax.text(2.1, hi_[0] + 0.015, hi_[1], fontsize=6.3, color=INK2, va="bottom")
        ax.set_xscale("log"); ax.set_xlim(2, 400); ax.set_ylim(-0.03, 1.0)
        ax.set_xticks([2.9, 9.8, 44.4, 233.4]); ax.set_xticklabels(["2.9\n(z = 3)", "9.8\n(z = 1)", "44.4\n(z = 0.3)", "233.4\n(z = 0.1)"])
        ax.minorticks_off(); ax.set_xlabel(t["eps"]); ax.set_title(title, loc="left"); grid_y(ax); panel(ax, letter)
    axs[0].set_ylabel(t["util"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7)
    save(fig, "fig08_dp", lang)


# ---------------------------------------------------------------- Figure 9: Zemberek
def fig_zemberek(lang):
    t = T[lang]; e6 = J("e6_h4_zemberek_me.json")["results"]
    rets = [("bm25", "BM25"), ("e5-small", "e5-small"), ("mosaic-embed", "MOSAIC-Embed")]
    vari = [("ham", t["raw"], GRAY_D, "o"), ("kok", t["root"], BLUE, "s"), ("morfem", t["morph"], ORANGE, "D")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, q, title, letter in ((axs[0], "cloze", t["zt"], "a"), (axs[1], "natural", t["zp"], "b")):
        for i, (rk, rl) in enumerate(rets):
            yb = len(rets) - 1 - i
            for j, (vk, vl, c, mk) in enumerate(vari):
                if vk not in e6[q][rk]: continue
                m, (lo, hi) = e6[q][rk][vk]["mrr@10"]
                yy = yb + (0.22 - 0.22 * j)
                ax.plot([lo, hi], [yy] * 2, color=c, lw=1.4); ax.plot(m, yy, marker=mk, ms=5, color=c, mec="white", mew=0.8, ls="none")
        ax.set_yticks(range(len(rets))); ax.set_yticklabels([r[1] for r in rets][::-1]); ax.set_xlim(0.6, 1.0)
        ax.set_ylim(-0.5, len(rets) - 0.5); ax.set_xlabel(t["zmrr"]); ax.set_title(title, loc="left"); grid_x(ax); panel(ax, letter)
    axs[1].legend(handles=[Line2D([], [], color=c, marker=mk, mec="white", label=vl) for _, vl, c, mk in vari],
                  loc="lower center", bbox_to_anchor=(-0.06, -0.45), ncol=3)
    save(fig, "fig09_zemberek", lang)


if __name__ == "__main__":
    for lang in ("en", "tr"):
        fig_router(lang); fig_retrieval(lang); fig_quant(lang); fig_h1(lang); fig_dp(lang); fig_zemberek(lang)
    print("ok")
