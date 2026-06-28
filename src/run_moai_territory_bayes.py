"""Bayesian cross-check: moai style cultural F_ST across the ethnohistoric named
territories (CLAN_BOUNDARY), reported only to confirm the geographic result does not
depend on dropping the clan model. NOT the analytical unit.

Mirrors the frequentist clan cross-check in run_moai.py but uses the same hierarchical
Bayesian (Balding-Nichols) estimator as the headline analysis, so the manuscript's
named-territory posterior traces to code.

Run (PyMC venv):  PYTHONPATH=src .venv/bin/python src/run_moai_territory_bayes.py
"""
import json

import bayes
import moai as moai_mod


def main():
    df = moai_mod.load()
    loci = moai_mod.select_loci(df)
    cl = df[df["CLAN_BOUNDARY"] != "NONE"].reset_index(drop=True)
    demes = cl["CLAN_BOUNDARY"].values
    n_terr = len(set(demes))
    counts = list(moai_mod.all_locus_counts(cl, demes, loci).values())
    idata = bayes.fst_posterior_multilocus(counts)
    s = bayes.fst_summary(idata)
    bf = bayes.bayes_factor_structure(counts)
    out = {"n_moai": int(len(cl)), "n_territories": int(n_terr),
           "fst_median": s["median"], "fst_hdi": [s["hdi_lo"], s["hdi_hi"]],
           "rhat": s["rhat"], "two_ln_bf": bf["two_ln_bf"], "evidence": bf["evidence"]}
    print(f"named territories (n={len(cl)}, {n_terr} territories):")
    print(f"  posterior F_ST = {s['median']:.3f} [{s['hdi_lo']:.3f}, {s['hdi_hi']:.3f}]  "
          f"R-hat={s['rhat']:.3f}")
    print(f"  2 ln BF = {bf['two_ln_bf']:+.1f} ({bf['evidence']})")
    json.dump(out, open("output/moai_territory_bayes.json", "w"), indent=2)
    print("wrote output/moai_territory_bayes.json")


if __name__ == "__main__":
    main()
