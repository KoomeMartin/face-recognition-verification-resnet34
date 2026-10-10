"""Generate the README results figure (light + dark variants).

    python scripts/make_figures.py

Outputs (written to ./assets):
    val-metrics-{light,dark}.svg   - validation classification accuracy and verification EER
                                     per epoch for the three main experiment stages

Curves are read from ``submission_artifacts/wandb_export.pkl`` (W&B run export). The
architecture diagram (assets/architecture.png) is hand-made and not generated here.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "s1": "#2a78d6",
        "s2": "#eb6834",
        "s3": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "s1": "#3987e5",
        "s2": "#d95926",
        "s3": "#199e70",
    },
}

# W&B run ids for each stage (see the experiment-history table in README.md)
STAGES = [
    ("8jsnzipf", "Stage 1 · GAP + softmax CE", "s1"),
    ("vm7vov1s", "Stage 2 · GDC + ArcFace", "s2"),
    ("8vwks9fe", "Stage 3 · GDC + softmax CE (final)", "s3"),
]
FINAL_ID = "8vwks9fe"


def load_runs():
    with open(os.path.join(ROOT, "submission_artifacts", "wandb_export.pkl"), "rb") as f:
        runs = pickle.load(f)
    out = {}
    for r in runs:
        rows = [h for h in r["history"] if h.get("valid_cls_acc") is not None]
        out[r["id"]] = {
            "epoch": [h["_step"] + 1 for h in rows],
            "acc": [h["valid_cls_acc"] for h in rows],
            "eer": [h["valid_eer"] for h in rows],
        }
    return out


def style_axis(ax, t):
    ax.set_facecolor(t["surface"])
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=10)
    ax.set_xlabel("Epoch", color=t["ink2"], fontsize=11)


def metrics_svg(t, path):
    runs = load_runs()
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})
    fig, (ax_acc, ax_eer) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=100)
    fig.patch.set_facecolor(t["surface"])

    for run_id, label, key in STAGES:
        r = runs[run_id]
        lw = 2.4 if run_id == FINAL_ID else 2
        z = 4 if run_id == FINAL_ID else 3
        ax_acc.plot(r["epoch"], r["acc"], color=t[key], lw=lw, label=label, zorder=z,
                    solid_capstyle="round")
        ax_eer.plot(r["epoch"], r["eer"], color=t[key], lw=lw, zorder=z, solid_capstyle="round")

    # Final model (best combined checkpoint, epoch 30)
    fin = runs[FINAL_ID]
    fe, fa, fr = fin["epoch"][-1], fin["acc"][-1], fin["eer"][-1]
    for ax, y in ((ax_acc, fa), (ax_eer, fr)):
        ax.plot(fe, y, "D", ms=9, color=t["s3"], mec=t["surface"], mew=2, zorder=6)
    ax_acc.annotate(f"{fa:.2f}%  final", (fe, fa), xytext=(-8, 12), textcoords="offset points",
                    ha="right", color=t["ink"], fontsize=10.5, fontweight="bold")
    ax_eer.annotate(f"{fr:.2f}%  final", (fe, fr), xytext=(-6, 14), textcoords="offset points",
                    ha="right", color=t["ink"], fontsize=10.5, fontweight="bold")

    # ArcFace reaches its lowest EER early, while its accuracy still lags
    arc = runs["vm7vov1s"]
    i = min(range(len(arc["eer"])), key=arc["eer"].__getitem__)
    ax_eer.plot(arc["epoch"][i], arc["eer"][i], "o", ms=8, color=t["s2"], mec=t["surface"],
                mew=2, zorder=6)
    ax_eer.annotate(f"{arc['eer'][i]:.2f}%  ArcFace, epoch {arc['epoch'][i]}",
                    (arc["epoch"][i], arc["eer"][i]), xytext=(4, -16),
                    textcoords="offset points", color=t["ink"], fontsize=10.5,
                    fontweight="bold")

    style_axis(ax_acc, t)
    ax_acc.set_xlim(0, 31)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_title("Classification accuracy (%)  ·  higher is better", loc="left",
                     color=t["ink2"], fontsize=11, pad=8)
    leg = ax_acc.legend(loc="lower right", frameon=False, fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    style_axis(ax_eer, t)
    ax_eer.set_xlim(0, 31)
    ax_eer.set_ylim(0, 10)
    ax_eer.set_title("Verification EER (%)  ·  lower is better (axis cut at 10)", loc="left",
                     color=t["ink2"], fontsize=11, pad=8)

    fig.suptitle("Validation metrics across experiment stages", x=0.012, ha="left",
                 color=t["ink"], fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for mode, t in THEMES.items():
        metrics_svg(t, os.path.join(ASSETS, f"val-metrics-{mode}.svg"))
    print("Figures written to", ASSETS)


if __name__ == "__main__":
    main()
