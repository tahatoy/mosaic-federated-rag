"""MOSAIC article and thesis figures (EN + TR).

Every value is read from ai/experiments/results/ (final configuration with MOSAIC-Embed).
Run:  python scripts/figures/figs.py      -> fig04..fig09 in EN and TR, fig_bench in TR (thesis Figure 4.3)
"""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("MOSAIC_RESULTS", os.path.join(HERE, "..", "..", "ai", "experiments", "results"))
OUT = os.environ.get("MOSAIC_FIG_OUT", HERE)
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
    """Result file relative to ai/experiments/results/ (q1/ files carry the prefix)."""
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name, lang):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}_{lang}.{ext}", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def grid_x(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def panel(ax, letter, x=-0.01):
    ax.text(x, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=8.5)


def thousands(n, lang):
    """1,019 in English; 1 019 (no-break space) in Turkish, where the point is the decimal separator."""
    s = f"{n:,}"
    return s if lang == "en" else s.replace(",", "\u00a0")


def pct(v):
    v = 100 * v
    return "100" if round(v, 1) == 100 else f"{v:.1f}"


T = {
    "en": dict(
        r_first="First router (live system)", r_new="Retrained router", r_acc="Correctly classified (%)",
        r_a="Templated test set:\nsame sources as training", r_b="Question sources\nnot used in training",
        r_ch="Chitchat", r_si="Simple", r_cx="Complex",
        r_rs="Simple:\nreport-style", r_ct="Simple:\ncross-branch\ntemplated", r_cp="Simple:\ncross-branch\nparaphrased",
        r_rc="Complex:\nreport-style",
        mrr="MRR@10 (federated setting, 95% CI)", templ="Templated (n = 600)", para="Paraphrased (n = 419)",
        bge="BGE-M3 (dense)",
        mem="GPU memory", speed="Generation speed", em_q="Cross-branch EM (same contexts)", gib="GiB", tps="Tokens/s",
        file="Model file", vram="GPU memory", em="EM (%)", nt="not\ntested",
        h1a="Templated questions (n = 500)", h1b="Paraphrased questions (n = 352)", h1c="Live end-to-end (n = 200, paraphrased)",
        emci="Exact match (%, 95% CI)",
        A="No retrieval", B="Asking branch only", C="MOSAIC (federated), MOSAIC-Embed", D="Central index, MOSAIC-Embed",
        L0="Qwen3-8B without retrieval", L1="MOSAIC, MOSAIC-Embed",
        g_base="Baselines", g_fed="MOSAIC (federated)", g_cen="Central index (upper bound)",
        eps="ε over three rounds (δ = 10⁻⁵, log scale)", util="MRR@10 (mean ± SD, 3 seeds)",
        cdp="Central DP (noise added once at HQ)", ldp="Local DP (noise added by each client)",
        nodp="No DP", base="Base e5-small (untrained)",
        zt="Templated questions (n = 600)", zp="Paraphrased questions (n = 419)",
        raw="Raw text", root="Root reduction", morph="Morpheme segmentation", zmrr="MRR@10 (95% CI)",
    ),
    "tr": dict(
        r_first="İlk yönlendirici (canlı sistem)", r_new="Yeniden eğitilen yönlendirici", r_acc="Doğru sınıflandırılan (%)",
        r_a="Kalıp test kümesi:\neğitimle aynı kaynak", r_b="Eğitimde kullanılmayan\nsoru kaynakları",
        r_ch="Gündelik", r_si="Basit", r_cx="Karmaşık",
        r_rs="Basit:\nrapor dili", r_ct="Basit:\nçapraz şube\nkalıp", r_cp="Basit:\nçapraz şube\nyeniden ifade",
        r_rc="Karmaşık:\nrapor dili",
        mrr="MRR@10 (federe düzen, %95 GA)", templ="Kalıp sorular (n = 600)", para="Yeniden ifade (n = 419)",
        bge="BGE-M3 (yoğun)",
        mem="Grafik bellek", speed="Üretim hızı", em_q="Çapraz şube EM (aynı bağlam)", gib="GiB", tps="Token/s",
        file="Model dosyası", vram="Grafik bellek", em="EM (%)", nt="sınanmadı",
        h1a="Kalıp sorular (n = 500)", h1b="Yeniden ifade edilmiş sorular (n = 352)", h1c="Canlı uçtan uca (n = 200, yeniden ifade)",
        emci="Tam eşleşme (%, %95 GA)",
        A="Bilgi getirme yok", B="Yalnız soran şube", C="MOSAIC (federe), MOSAIC-Embed", D="Merkezi dizin, MOSAIC-Embed",
        L0="Bilgi getirmesiz Qwen3-8B", L1="MOSAIC, MOSAIC-Embed",
        g_base="Temel koşullar", g_fed="MOSAIC (federe)", g_cen="Merkezi dizin (üst sınır)",
        eps="Üç turda ε (δ = 10⁻⁵, logaritmik ölçek)", util="MRR@10 (ortalama ± SS, 3 tohum)",
        cdp="Merkezi DP (gürültü HQ'da bir kez)", ldp="Yerel DP (gürültü her istemcide)",
        nodp="DP yok", base="Taban e5-small (eğitimsiz)",
        zt="Kalıp sorular (n = 600)", zp="Yeniden ifade edilmiş sorular (n = 419)",
        raw="Ham metin", root="Kök indirgeme", morph="Morfem bölütleme", zmrr="MRR@10 (%95 GA)",
    ),
}


# ---------------------------------------------------------------- Figure 4: router
def fig_router(lang):
    """(a) per-class recall on the template test set (E19); (b) accuracy on question sources not used in training (E18).
    Classifier decisions without the rules of Section 2.6; retrained router = mean of three seeds."""
    t = T[lang]; e18 = J("q1/e18_router_v2.json")["models"]; e19 = J("q1/e19_router_per_class.json")
    v1, runs = e18["bert_v1_live"], e18["bert_v2_runs"]["raw"]
    mean = lambda key: st.mean(r[key]["acc"] for r in runs)
    n_cls = e19["n"] // 3
    a_rows = [(f"{t['r_ch']}\n(n = {thousands(n_cls, lang)})", e19["bert_router_best"]["raw"]["recall"]["chitchat"], e19["bert_router_v2"]["raw"]["recall"]["chitchat"]),
              (f"{t['r_si']}\n(n = {thousands(n_cls, lang)})", e19["bert_router_best"]["raw"]["recall"]["simple"], e19["bert_router_v2"]["raw"]["recall"]["simple"]),
              (f"{t['r_cx']}\n(n = {thousands(n_cls, lang)})", e19["bert_router_best"]["raw"]["recall"]["complex"], e19["bert_router_v2"]["raw"]["recall"]["complex"])]
    b_keys = [("report_simple_heldout", t["r_rs"]), ("crossbranch_cloze", t["r_ct"]),
              ("crossbranch_paraphrased", t["r_cp"]), ("report_complex_heldout", t["r_rc"])]
    b_rows = [(f"{lab}\n(n = {v1[k]['n']})", v1[k]["acc"], mean(k)) for k, lab in b_keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[3, 4], wspace=0.08), sharey=True)
    for ax, rows in ((a1, a_rows), (a2, b_rows)):
        for i, (lab, old, new) in enumerate(rows):
            ax.bar(i - 0.19, 100 * old, 0.36, color=GRAY, edgecolor="white", lw=0.8, hatch="////")
            ax.bar(i + 0.19, 100 * new, 0.36, color=BLUE, edgecolor="white", lw=0.8)
            ax.text(i - 0.19, 100 * old + 2, pct(old), ha="center", va="bottom", fontsize=6.8)
            ax.text(i + 0.19, 100 * new + 2, pct(new), ha="center", va="bottom", fontsize=6.8)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=6.9, color=INK2)
        ax.tick_params(axis="x", length=0); grid_y(ax); ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    a2.axvline(2.5, color=INK2, lw=0.9, ls="--"); a2.spines["left"].set_visible(False); a2.tick_params(axis="y", length=0)
    a1.set_ylabel(t["r_acc"])
    a1.set_title(t["r_a"], loc="left"); a2.set_title(t["r_b"], loc="left"); panel(a1, "a"); panel(a2, "b")
    fig.legend(handles=[Patch(fc=GRAY, ec="white", hatch="////", label=t["r_first"]), Patch(fc=BLUE, label=t["r_new"])],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6, columnspacing=1.6)
    save(fig, "fig04_router", lang)


# ---------------------------------------------------------------- Figure 5: retrieval benchmark
def fig_retrieval(lang):
    t = T[lang]; e1 = J("q1/e1_retrieval_me.json")["methods"]
    names = {"bm25": "BM25", "e5-small": "e5-small", "bge-m3": t["bge"], "turkembed4retrieval": "TurkEmbed4Retrieval",
             "mosaic-embed": "MOSAIC-Embed", "hybrid-rrf": "BM25 + MOSAIC-Embed (RRF)",
             "e5-small+ce": "e5-small + CE", "mosaic-embed+ce": "MOSAIC-Embed + CE", "hybrid-rrf+ce": "BM25 + MOSAIC-Embed (RRF) + CE"}
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
    """File size and GPU memory: GGUF files and llama-server with context cache (Phase 6 of the thesis, 50 questions);
    EM: untuned Gemma-3-4B on identical MOSAIC contexts, 500 cross-branch questions (E2, quantization arms)."""
    t = T[lang]; Q = J("q1/e2_generation_me_quant.json")["sonuclar"]["cloze"]
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
    M = J("q1/e2_generation_me_full.json")["sonuclar"]     # final configuration, MOSAIC-Embed
    live = J("q1/e5_e2e_judge_v3.json")["summary"]           # live system, final configuration
    panels = [
        (t["h1a"], [(t["A"], M["cloze"]["A"]["em"], GRAY, ""), (t["B"], M["cloze"]["B"]["em"], GRAY, ""),
                    (t["C"], M["cloze"]["C"]["em"], BLUE, ""), (t["D"], M["cloze"]["D_me"]["em"], ORANGE, "////")]),
        (t["h1b"], [(t["A"], M["natural"]["A"]["em"], GRAY, ""), (t["C"], M["natural"]["C"]["em"], BLUE, ""),
                    (t["D"], M["natural"]["D_me"]["em"], ORANGE, "////")]),
        (t["h1c"], [(t["L0"], live["norag_em"], GRAY, ""), (t["L1"], live["em"], BLUE, "")]),
    ]
    heights = [len(p[1]) for p in panels]
    fig, axs = plt.subplots(3, 1, figsize=(W, 3.3), sharex=True, gridspec_kw=dict(height_ratios=heights, hspace=0.55))
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
                  ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.2), handlelength=1.4)
    save(fig, "fig07_crossbranch", lang)


# ---------------------------------------------------------------- Figure 8: privacy-utility
def fig_dp(lang):
    """ε from Rényi accounting over the orders alpha = 1.1, 1.2, ..., 10.9 and 11, 12, ..., 256, δ = 1e-5."""
    t = T[lang]; d = J("q1/e3_fl_embedding_qp.json")  # query–passage pairs
    g = {}
    for r in d["runs"]:
        g.setdefault(r["name"] + (f"|{r['z']}" if "z" in r else ""), []).append(r)
    zs = sorted({r["z"] for r in d["runs"] if "z" in r}, reverse=True)
    eps = [g[f"dp_central_K10_lora|{z}"][0]["epsilon"] for z in zs]
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
        ax.set_xticks(eps); ax.set_xticklabels([f"{e:.1f}\n(z = {z:g})" for e, z in zip(eps, zs)])
        ax.minorticks_off(); ax.set_xlabel(t["eps"]); ax.set_title(title, loc="left"); grid_y(ax); panel(ax, letter)
    axs[0].set_ylabel(t["util"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7)
    save(fig, "fig08_dp", lang)


# ---------------------------------------------------------------- Figure 9: Zemberek
def fig_zemberek(lang):
    t = T[lang]; e6 = J("q1/e6_h4_zemberek_me.json")["results"]
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


# ---------------------------------------------------------------- Thesis Figure 4.3: general-ability benchmarks (TR)
def fig_bench(lang="tr"):
    q_c, g_c = J("bench_chatml_qwen.json"), J("bench_chatml_gemma.json")
    q_n, g_n = J("bench_native_qwen.json"), J("bench_native_gemma.json")
    tm = J("q1/e10_tr_mmlu_summary.json")
    conds = [("Qwen3-8B, ChatML", q_c, tm["qwen_chatml"]["acc"], "sentiment", BLUE, ""),
             ("Gemma-3-4B, ChatML", g_c, tm["gemma_chatml"]["acc"], "sentiment", ORANGE, ""),
             ("Qwen3-8B, kendi şablonu", q_n, tm["qwen_native"]["acc"], "sentiment_binary", BLUE, "////"),
             ("Gemma-3-4B, kendi şablonu", g_n, tm["gemma_native"]["acc"], "sentiment_binary", ORANGE, "////")]
    groups = ["MMLU-TR doğruluk\n(n = 500)", f"TR-MMLU doğruluk\n(n = {thousands(tm['qwen_chatml']['n'], 'tr')})",
              "Duygu doğruluk\n(n = 500)", "Duygu Macro F1 × 100\n(n = 500)"]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    wb = 0.2
    for j, (lab, d, trm, sk, c, h) in enumerate(conds):
        vals = [d["tr_mmlu"]["accuracy"], trm, d[sk]["accuracy"], d[sk]["macro_f1"]]
        for i, v in enumerate(vals):
            x = i + (j - 1.5) * wb
            ax.bar(x, 100 * v, wb * 0.92, color=c, edgecolor="white", lw=0.8, hatch=h, label=lab if i == 0 else None)
            ax.text(x, 100 * v + 1.2, f"{100 * v:.1f}", ha="center", va="bottom", fontsize=5.9, rotation=0)
    ax.hlines(25, -0.45, 0.45, color=INK2, lw=0.9, ls=":"); ax.text(0.47, 25, "%25", va="center", fontsize=6.3, color=INK2)
    ax.hlines(20, 0.55, 1.45, color=INK2, lw=0.9, ls=":"); ax.text(1.47, 20, "%20", va="center", fontsize=6.3, color=INK2)
    base = q_n["sentiment_binary"]["majority_baseline_acc"]
    ax.hlines(100 * base, 1.55, 2.45, color=INK2, lw=0.9, ls="--"); ax.text(2.47, 100 * base, f"%{100 * base:.1f}", va="center", fontsize=6.3, color=INK2)
    ax.set_xticks(range(4)); ax.set_xticklabels(groups, fontsize=7); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100); ax.set_ylabel("Yüzde (%)"); grid_y(ax)
    h_, l_ = ax.get_legend_handles_labels()
    h_ += [Line2D([], [], color=INK2, ls=":", lw=0.9), Line2D([], [], color=INK2, ls="--", lw=0.9)]
    l_ += ["Rastgele tahmin düzeyi", "Duygu çoğunluk sınıfı tabanı"]
    ax.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.8, handlelength=1.5, columnspacing=1.2)
    save(fig, "fig_bench", lang)


if __name__ == "__main__":
    for lang in ("en", "tr"):
        fig_router(lang); fig_retrieval(lang); fig_quant(lang); fig_h1(lang); fig_dp(lang); fig_zemberek(lang)
    fig_bench("tr")
    print("ok")
