"""How demic structure emerges by combining many individually-variable dimensions.

The moai case in three panels:
  A. Within every community each style dimension is variable, and the states recur
     across communities -- no single dimension cleanly separates places.
  B. Estimated on its own, each dimension's between-community F_ST is uncertain
     (wide intervals, most reaching zero); pooling the seven into one multilocus
     genotype gives a decisive, well-constrained signal.
  C. The combined genotype is spatially patterned: statues within a few kilometers
     are more alike than chance, with no difference beyond.

Reads output/bayes_results.json (per-locus + multilocus posteriors) and
output/moai_spatial.txt (distance-binned style distance). DEM-free, system python3.
Run:  python3 src/make_demic_structure_figure.py
"""
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import moai as moai_mod

PALETTE = ["#d7191c", "#2c7bb6", "#fdae61", "#1a9641", "#7b3294"]


def _panel_composition(ax):
    df = moai_mod.load()
    demes = moai_mod.primary_demes(df)
    hp = moai_mod._clean(df["HEAD_PLAN_SHAPE"])
    d = pd.Series(demes, index=df.index)
    t = pd.crosstab(d, hp)
    lon = pd.Series(df["longitude"].values, index=df.index).groupby(d).mean()
    t = t.loc[lon.sort_values().index]
    comp = t.div(t.sum(axis=1), axis=0)
    states = list(comp.columns)
    x = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for st, col in zip(states, PALETTE):
        ax.bar(x, comp[st].values, bottom=bottom, color=col, edgecolor="white",
               width=0.8, label=str(st).replace("_", " "))
        bottom += comp[st].values
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i+1}" for i in range(len(comp))], fontsize=8)
    ax.set_xlabel("moai community (west → east)", fontsize=9)
    ax.set_ylabel("composition", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("A. Each dimension is variable within every community\n"
                 "(head plan shape; states recur across communities)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7, ncol=2, loc="upper center", framealpha=0.9)


def _panel_perlocus(ax):
    m = json.load(open("output/bayes_results.json"))["proxies"]["moai"]
    marg = m["marginal"]
    rows = sorted(marg.items(), key=lambda kv: kv[1]["median"])
    names = [k.replace("_", " ").title() for k, _ in rows]
    med = np.array([v["median"] for _, v in rows])
    lo = np.array([v["hdi_lo"] for _, v in rows])
    hi = np.array([v["hdi_hi"] for _, v in rows])
    y = np.arange(len(rows))
    ax.hlines(y, lo, hi, color="#888888", lw=2)
    ax.scatter(med, y, s=34, color="#333333", zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    # combined multilocus
    h = m["headline"]
    ax.axvspan(h["hdi_lo"], h["hdi_hi"], color="#1a9641", alpha=0.18, zorder=0)
    ax.axvline(h["median"], color="#1a9641", lw=2)
    ax.axvline(0, color="#bbbbbb", lw=1, ls=":")
    ax.set_xlabel("between-community F$_{ST}$ (posterior median, 95% HDI)", fontsize=9)
    ax.set_title("B. Single dimensions are individually uncertain;\n"
                 "the combined multilocus genotype is decisive", fontsize=9.5, loc="left")
    ax.annotate(f"combined (7 loci)\nF$_{{ST}}$ = {h['median']:.2f}\n2 ln BF = +14 (decisive)",
                (0.20, 4.4), fontsize=8, color="#0b6b2e", ha="left", va="center")
    ax.set_xlim(left=-0.01)


def _panel_decay(ax):
    text = open("output/moai_spatial.txt").read()
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s*([\d\-]+(?:-\d+)?)\s*km\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows.append((m.group(1) + " km", float(m.group(3)), float(m.group(4))))
    labels = [r[0] for r in rows]
    obs = np.array([r[1] for r in rows])
    null = np.array([r[2] for r in rows])
    x = np.arange(len(rows))
    ax.plot(x, null, "o--", color="#999999", lw=1.5,
            label="expected if style independent of location")
    ax.plot(x, obs, "o-", color="#d7191c", lw=2, label="observed")
    closer = obs < null
    ax.fill_between(x, obs, null, where=closer, color="#d7191c", alpha=0.15,
                    interpolate=True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlim(-0.3, len(rows) - 0.6)
    ax.set_xlabel("distance between statues", fontsize=9)
    ax.set_ylabel("mean style distance\n(lower = more alike)", fontsize=9)
    ax.set_title("C. Combined, nearby statues are more alike than chance\n"
                 "(spatial structure at a few-kilometer grain)", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main(path="figures/moai_demic_structure.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    _panel_composition(axes[0])
    _panel_perlocus(axes[1])
    _panel_decay(axes[2])
    fig.suptitle("How demic structure emerges: individually variable dimensions, "
                 "jointly patterned in space (moai style)", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
