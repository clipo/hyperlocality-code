"""Bayesian re-analysis driver for the four hyperlocality proxies.

Reports, for each proxy, the model-based replacements for the frequentist battery:
posterior cultural F_ST (median + 95% credible interval), the structure-vs-panmixia
Bayes factor, per-attribute/per-locus posteriors, leave-one-deme-out posteriors,
the within-cluster posterior (mata'a), and the Bayesian isolation-by-distance slope.
Each headline Bayes factor is also reported under a second prior on F as a
sensitivity check. The Nei G_ST point estimate and its panmixia-null mean are
printed alongside for concordance (they are demoted to the Supplementary Material).

Writes a human report per proxy to output/bayes_<proxy>.txt, a cross-proxy table to
output/bayes_summary.txt, and a machine-readable output/bayes_results.json that the
figures and the manuscript draw from (the single source of truth for every number).

Run:  PYTHONPATH=src .venv/bin/python src/run_bayes.py
"""
import json
import sys

import numpy as np

import bayes
import popgen
import spatial
import tables_io
import umu as umu_mod
import pukao as pukao_mod
import moai as moai_mod
import harden  # reuse marginalize() for the mata'a attribute splits

# A second prior on F for the sensitivity check: Beta(1,3) puts more mass on small
# differentiation, a sterner test for the positive proxies. The F_ST credible
# interval (reported throughout) is far less prior-sensitive than the Bayes factor.
ALT_PRIOR = ("beta", 1.0, 3.0)

SW = ["Ahu Tautira", "Orongo", "Orito", "Rano Kau", "Vinapu"]
PARCELS = ["Parcel 6", "Parcel 7", "Parcel 8", "Parcel 9", "Parcel 10", "Parcel 11"]

SUMMARY = []          # rows for the cross-proxy table
RESULTS = {}          # full structured results -> bayes_results.json


class _Tee:
    """Write to the console and the per-proxy report file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _post_only(counts_or_list):
    """Posterior F_ST summary (no Bayes factor) for one (multi)locus matrix."""
    idata = (bayes.fst_posterior_multilocus(counts_or_list)
             if isinstance(counts_or_list, list)
             else bayes.fst_posterior(counts_or_list))
    s = bayes.fst_summary(idata)
    gst = (popgen.gst_multilocus(counts_or_list) if isinstance(counts_or_list, list)
           else popgen.gst(counts_or_list)[0])
    s["gst"] = float(gst)
    return s


def _fst_line(label, counts_or_list, gst_value, gst_null=None):
    """Posterior F_ST + Bayes factor for one (multi)locus test, with G_ST concord."""
    s = _post_only(counts_or_list)
    bf = bayes.bayes_factor_structure(counts_or_list)
    rec = {**s, "gst": float(gst_value),
           "gst_null": (float(gst_null) if gst_null is not None else None),
           "two_ln_bf": bf["two_ln_bf"], "log10_bf": bf["log10_bf"],
           "chain_sd": bf["two_ln_bf_chain_sd"], "evidence": bf["evidence"]}
    print(f"  {label}")
    concord = f"   [Nei G_ST {gst_value:.4f}" + (
        f", null {gst_null:.4f}" if gst_null is not None else "") + "]"
    print(f"    F_ST posterior: median={s['median']:.4f}  "
          f"95% HDI=[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]" + concord)
    print(f"    Bayes factor (structure vs panmixia): 2 ln BF = {bf['two_ln_bf']:+.1f}"
          f"  (log10 BF = {bf['log10_bf']:+.1f}; chain sd {bf['two_ln_bf_chain_sd']:.2f})"
          f"  -> {bf['evidence']}")
    print(f"    diagnostics: R-hat={s['rhat']:.3f}  ESS={s['ess']:.0f}")
    return rec, bf


def _prior_sensitivity(label, counts_or_list, base_bf):
    idata = (bayes.fst_posterior_multilocus(counts_or_list, f_prior=ALT_PRIOR)
             if isinstance(counts_or_list, list)
             else bayes.fst_posterior(counts_or_list, f_prior=ALT_PRIOR))
    s = bayes.fst_summary(idata)
    bf = bayes.bayes_factor_structure(counts_or_list, f_prior=ALT_PRIOR)
    print(f"  prior sensitivity ({label}, F~Beta(1,3)):")
    print(f"    F_ST median={s['median']:.4f} [{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]"
          f"  |  2 ln BF = {bf['two_ln_bf']:+.1f} ({bf['evidence']})")
    print(f"    (uniform-prior 2 ln BF was {base_bf['two_ln_bf']:+.1f})")
    return {"median": s["median"], "hdi_lo": s["hdi_lo"], "hdi_hi": s["hdi_hi"],
            "two_ln_bf": bf["two_ln_bf"], "evidence": bf["evidence"],
            "uniform_two_ln_bf": base_bf["two_ln_bf"]}


def _ibd(Dcomp, Dgeo, region=None):
    out = bayes.ibd_regression(Dcomp, Dgeo, region=region)
    tag = "distance | region" if region is not None else "distance"
    print(f"    IBD {tag:16s}: slope={out['slope_median']:+.3f}  "
          f"95% HDI=[{out['slope_hdi_lo']:+.3f}, {out['slope_hdi_hi']:+.3f}]  "
          f"P(slope>0)={out['p_positive']:.2f}")
    return out


def _summary_row(proxy, rec):
    SUMMARY.append({"proxy": proxy, "median": rec["median"], "lo": rec["hdi_lo"],
                    "hi": rec["hdi_hi"], "two_ln_bf": rec["two_ln_bf"],
                    "evidence": rec["evidence"], "gst": rec["gst"]})


def banner(title):
    print("=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
def report_mataa():
    banner("MATA'A  (Bayesian cultural F_ST; published Canvas count tables)")
    R = RESULTS.setdefault("mataa", {})
    lw, _ = tables_io.stem_length_width()
    ss, _ = tables_io.stem_shoulder_shape()
    Clw = lw.values.astype(float)
    Css = ss.values.astype(float)

    print("\n[headline F_ST -- stem length x width, single 14-class locus]")
    g = popgen.gst(Clw)[0]
    _, _, gnull, _ = popgen.gst_permutation(Clw, n_perm=2999)
    rec_lw, bf_lw = _fst_line("stem length x width", Clw, g, gnull)
    R.setdefault("headline", {})["lengthwidth"] = rec_lw
    _summary_row("mata'a stem L x W", rec_lw)
    R["prior_sensitivity_lw"] = _prior_sensitivity("stem L x W", Clw, bf_lw)

    print("\n[headline F_ST -- stem shape x shoulder shape, single 6-class locus]")
    g2 = popgen.gst(Css)[0]
    _, _, gnull2, _ = popgen.gst_permutation(Css, n_perm=2999)
    rec_ss, _ = _fst_line("stem shape x shoulder shape", Css, g2, gnull2)
    R["headline"]["shapeshoulder"] = rec_ss
    _summary_row("mata'a shape x shoulder", rec_ss)

    print("\n[per-attribute marginal F_ST (each attribute as one locus)]")
    R["marginal"] = {}
    for mat, splits in [(lw, [("stem length", lambda c: c[0]),
                              ("stem width", lambda c: c[1:])]),
                        (ss, [("stem shape", lambda c: c[:-1]),
                              ("shoulder shape", lambda c: c[-1])])]:
        for attr, key in splits:
            m = harden.marginalize(mat, key).values.astype(float)
            s = _post_only(m)
            R["marginal"][attr] = s
            print(f"    {attr:14s}: F_ST median={s['median']:.4f} "
                  f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]  (Nei {s['gst']:.4f})")

    print("\n[within-SW-cluster F_ST -- the load-bearing hyperlocality test]")
    sub = lw.loc[[c for c in SW if c in lw.index]]
    Csub = sub.values.astype(float)
    span = spatial.distance_matrix(list(sub.index))[np.triu_indices(len(sub), 1)].max()
    print(f"  {len(sub)} SW sites within {span:.1f} km of one another")
    g3 = popgen.gst(Csub)[0]
    _, _, gnull3, _ = popgen.gst_permutation(Csub, n_perm=2999)
    rec_wc, _ = _fst_line("within-cluster stem L x W", Csub, g3, gnull3)
    rec_wc["span_km"] = float(span)
    R["within_cluster"] = rec_wc
    # The full 14-class scheme is sparse over five small demes, so its Bayes factor
    # is underpowered; the within-cluster signal is clearer on the coarser marginals.
    print("  within-cluster by attribute (coarser, better powered):")
    R["within_cluster_marginal"] = {}
    for attr, key in [("stem length", lambda c: c[0]), ("stem width", lambda c: c[1:])]:
        m = harden.marginalize(sub, key).values.astype(float)
        gg = popgen.gst(m)[0]
        _, _, gnm, _ = popgen.gst_permutation(m, n_perm=2999)
        rec_m, _ = _fst_line(f"within-cluster {attr}", m, gg, gnm)
        R["within_cluster_marginal"][attr] = rec_m
    print("  leave-one-site-out (posterior F_ST):")
    R["loo"] = {}
    for drop in sub.index:
        keep = [n for n in sub.index if n != drop]
        s = _post_only(sub.loc[keep].values.astype(float))
        R["loo"][drop] = s
        print(f"    -{drop:12s} F_ST median={s['median']:.4f} "
              f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]")

    print("\n[scale-free outline classes -- negative control, expect panmixia]")
    try:
        from classify import build
        _, matA, _ = build(n_states=3)
        Ca = matA.values.astype(float)
        gA = popgen.gst(Ca)[0]
        _, _, gnullA, _ = popgen.gst_permutation(Ca, n_perm=2999)
        rec_out, _ = _fst_line("outline classes (3x3)", Ca, gA, gnullA)
        R["outline"] = rec_out
        _summary_row("mata'a outline (control)", rec_out)
    except Exception as e:
        print(f"  [outline classes skipped: {e}]")

    print("\n[isolation by distance -- stem length x width, secondary]")
    names_g = [n for n in lw.index if spatial.has_coords(n)]
    Cg = lw.loc[names_g].values.astype(float)
    Dgeo = spatial.distance_matrix(names_g)
    Dcomp = popgen.neiman_d2(Cg)
    region = np.array([0 if n in PARCELS else 1 for n in names_g])
    Dregion = (region[:, None] != region[None, :]).astype(float)
    R["ibd"] = {"plain": _ibd(Dcomp, Dgeo),
                "region": _ibd(Dcomp, Dgeo, region=Dregion)}
    print("    NB pairwise distances are non-independent; IBD is a secondary check.")


def report_umu():
    banner("UMU PAE  (Bayesian cultural F_ST; McCoy 1978 Table 1)")
    R = RESULTS.setdefault("umu", {})
    df = umu_mod.load()
    C = umu_mod.counts(df)
    demes = list(df["quadrangle"])
    print(f"\n{C.shape[0]} survey quadrangles, {C.shape[1]} oven styles, "
          f"n={int(C.sum())} classifiable ovens")

    print("\n[headline F_ST -- oven style, single 4-class locus]")
    g = popgen.gst(C)[0]
    _, _, gnull, _ = popgen.gst_permutation(C, n_perm=2999)
    rec, bf = _fst_line("oven style", C, g, gnull)
    R["headline"] = rec
    _summary_row("umu oven style", rec)
    R["prior_sensitivity"] = _prior_sensitivity("oven style", C, bf)

    print("\n[per-style marginal F_ST (style vs rest)]")
    R["marginal"] = {}
    tot = C.sum(1, keepdims=True)
    for k, st in enumerate(umu_mod.STYLES):
        binary = np.column_stack([C[:, k], tot[:, 0] - C[:, k]])
        s = _post_only(binary)
        R["marginal"][st] = s
        print(f"    {st:12s}: F_ST median={s['median']:.4f} "
              f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]  (Nei {s['gst']:.4f})")

    print("\n[leave-one-quadrangle-out (posterior F_ST)]")
    R["loo"] = {}
    for i, q in enumerate(demes):
        s = _post_only(np.delete(C, i, axis=0))
        R["loo"][q] = s
        print(f"    -{q:14s} F_ST median={s['median']:.4f} "
              f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]")

    print("\n[isolation by distance -- exploratory, 5 demes, map-read coords]")
    Dgeo = umu_mod.distance_km(df)
    Dcomp = popgen.neiman_d2(C)
    R["ibd"] = {"plain": _ibd(Dcomp, Dgeo)}
    print("    NB 10 non-independent pairs; exploratory only.")


def report_moai():
    banner("MOAI  (Bayesian multilocus cultural F_ST; geographic spatial-cluster demes)")
    R = RESULTS.setdefault("moai", {})
    df = moai_mod.load()
    demes = moai_mod.primary_demes(df)
    loci = moai_mod.select_loci(df)
    counts = moai_mod.all_locus_counts(df, demes, loci)
    clist = list(counts.values())
    print(f"\n{len(df)} ahu-placed moai, {len(set(demes))} spatial clusters "
          f"(complete linkage k={moai_mod.PRIMARY_K}), {len(loci)} style loci")

    print("\n[headline multilocus F_ST]")
    g = moai_mod.multilocus_gst(counts)
    _, _, gnull, _ = moai_mod.multilocus_permutation(df, demes, loci, n_perm=999)
    rec, bf = _fst_line("7-locus style", clist, g, gnull)
    R["headline"] = rec
    _summary_row("moai style (k=6)", rec)
    R["prior_sensitivity"] = _prior_sensitivity("moai style", clist, bf)

    print("\n[per-locus F_ST]")
    R["marginal"] = {}
    for a, m in counts.items():
        s = _post_only(m)
        R["marginal"][a] = s
        print(f"    {a:24s}: F_ST median={s['median']:.4f} "
              f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]  (Nei {s['gst']:.4f})")

    print("\n[leave-one-cluster-out (posterior multilocus F_ST)]")
    R["loo"] = {}
    for d in sorted(set(demes)):
        mask = np.asarray(demes) != d
        cl = list(moai_mod.all_locus_counts(
            df[mask].reset_index(drop=True), np.asarray(demes)[mask], loci).values())
        s = _post_only(cl)
        R["loo"][str(d)] = s
        print(f"    -cluster {d:<4} F_ST median={s['median']:.4f} "
              f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]")

    print("\n[resolution robustness -- spatial clusters at k=6,8,10]")
    R["resolution"] = {}
    for k in (6, 8, 10):
        sd = moai_mod.spatial_demes(df, k=k, method="complete")
        keep = [d for d in set(sd) if (sd == d).sum() >= 5]
        mask = np.isin(sd, keep)
        cl = list(moai_mod.all_locus_counts(
            df[mask].reset_index(drop=True), sd[mask], loci).values())
        s = _post_only(cl)
        s["n_demes"], s["n"] = len(keep), int(mask.sum())
        R["resolution"][f"k{k}"] = s
        print(f"    k={k:<2} ({len(keep)} demes, n={int(mask.sum())}): "
              f"F_ST median={s['median']:.4f} [{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]")

    print("\n[isolation by distance -- cluster centroids, secondary]")
    Dgeo = moai_mod.deme_distance_km(df, demes)
    Dcomp = moai_mod.compositional_distance(counts)
    R["ibd"] = {"plain": _ibd(Dcomp, Dgeo)}


def report_pukao():
    banner("PUKAO  (Bayesian multilocus cultural F_ST; negative/underpowered proxy)")
    R = RESULTS.setdefault("pukao", {})
    df = pukao_mod.load()
    R["thresholds"] = {}
    for thr in (1500.0, 1000.0, 2000.0):
        demes = pukao_mod.deme_labels(df, threshold_m=thr)
        counts = pukao_mod.all_locus_counts(df, demes)
        clist = list(counts.values())
        n_demes = len(set(demes))
        print(f"\n[single-linkage cut {thr:.0f} m -> {n_demes} demes]")
        g = pukao_mod.multilocus_gst(counts)
        _, _, gnull, _ = pukao_mod.multilocus_permutation(df, demes, n_perm=999)
        rec, bf = _fst_line(f"{len(pukao_mod.LOCI)}-locus style", clist, g, gnull)
        rec["n_demes"] = n_demes
        R["thresholds"][f"{int(thr)}m"] = rec
        if thr == 1500.0:
            R["headline"] = rec
            _summary_row("pukao style (1500 m)", rec)
            R["prior_sensitivity"] = _prior_sensitivity("pukao style", clist, bf)
            print("\n  [per-locus F_ST at 1500 m]")
            R["marginal"] = {}
            for a, m in counts.items():
                s = _post_only(m)
                R["marginal"][a] = s
                print(f"    {a:12s}: F_ST median={s['median']:.4f} "
                      f"[{s['hdi_lo']:.4f}, {s['hdi_hi']:.4f}]  (Nei {s['gst']:.4f})")
            Dgeo = pukao_mod.deme_distance_km(df, demes)
            Dcomp = pukao_mod.compositional_distance(counts)
            print("\n  [isolation by distance -- secondary]")
            R["ibd"] = {"plain": _ibd(Dcomp, Dgeo)}


def write_summary(path="output/bayes_summary.txt"):
    lines = ["=" * 78, "BAYESIAN CULTURAL F_ST -- CROSS-PROXY SUMMARY", "=" * 78,
             f"{'proxy':<28}{'F_ST median [95% HDI]':<26}{'2 ln BF':>9}  evidence",
             "-" * 78]
    for r in SUMMARY:
        hdi = f"{r['median']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]"
        lines.append(f"{r['proxy']:<28}{hdi:<26}{r['two_ln_bf']:>+9.1f}  {r['evidence']}")
    lines += ["-" * 78,
              "Kass & Raftery (1995): |2 ln BF| 2-6 positive, 6-10 strong, >10 very strong.",
              "Nei G_ST point estimates (frequentist concordance, demoted to SM):"]
    for r in SUMMARY:
        lines.append(f"  {r['proxy']:<28} Nei G_ST = {r['gst']:.4f}")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nwrote {path}")


def write_json(path="output/bayes_results.json"):
    payload = {"seed": bayes.SEED, "draws": bayes.DRAWS, "tune": bayes.TUNE,
               "chains": bayes.CHAINS, "smc_draws": bayes.SMC_DRAWS,
               "alt_prior": "Beta(1,3)", "proxies": RESULTS, "summary": SUMMARY}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


PROXIES = {"mataa": report_mataa, "umu": report_umu,
           "moai": report_moai, "pukao": report_pukao}


if __name__ == "__main__":
    which = sys.argv[1:] or list(PROXIES)
    for name in which:
        if name in PROXIES:
            out = f"output/bayes_{name}.txt"
            with open(out, "w") as f:
                old = sys.stdout
                sys.stdout = _Tee(old, f)
                try:
                    PROXIES[name]()
                finally:
                    sys.stdout = old
            print(f"wrote {out}")
    write_summary()
    write_json()
