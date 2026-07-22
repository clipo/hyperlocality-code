"""Bayesian cultural-F_ST analysis for the hyperlocality proxies.

The Bayesian counterpart of popgen.py. It replaces the three frequentist objects
the paper used to report -- a Nei G_ST point estimate, a panmixia permutation
p-value, and a bootstrap interval -- with three model-based quantities that come
out of one hierarchical model:

  * posterior median F_ST + 95% credible interval   (was: point estimate + CI)
  * Bayes factor, structure vs panmixia             (was: permutation p-value)
  * posterior distance-decay slope                   (was: Mantel r + p)

Model (Balding & Nichols 1995; the cultural reading follows the same
classes-as-alleles / assemblages-as-demes move as popgen.py). For one locus with
K classes across S demes:

    pi  ~ Dirichlet(1, ..., 1)              island-wide class frequencies
    F   ~ Uniform(0, 1)                     cultural F_ST (the BN theta); a = (1-F)/F
    x_i ~ DirichletMultinomial(n_i, a * pi) deme i's counts

F is exactly the between-deme differentiation parameter, estimated directly. The
per-deme frequencies are marginalized analytically inside the Dirichlet-Multinomial,
so the only unknowns are F and pi. Unlike Nei's G_ST this is not biased upward in
small/unequal samples -- the partial pooling is in the model -- so we no longer need
the "bias cancels in the null" argument; the small-sample uncertainty is the width
of the F posterior.

Panmixia (M0) is the F -> 0 limit: every deme shares one pi and the counts are
plain Multinomial. Because M0 is nested in M1 (F free), the marginal likelihoods
are directly comparable and BF10 = mL(M1)/mL(M0) measures evidence for structure.
We estimate each model's marginal likelihood with sequential Monte Carlo
(pm.sample_smc), report 2 ln BF on the Kass & Raftery (1995) scale, and check
prior sensitivity in run_bayes.py.

Multilocus proxies (pukao, moai, and the two mata'a cross-classifications read as
two attributes) share one F across loci, with a separate pi per locus -- the
Bayesian analogue of Nei's multilocus G_ST.
"""
import contextlib
import io
import logging

import numpy as np
import pymc as pm
import arviz as az

# PyMC/SMC emit "Initializing ... / Sampling N chains" lines via the logging
# module, which bypasses stdout redirection; quiet them so run_bayes.py output
# stays readable. Diagnostics (R-hat, ESS) are computed explicitly instead.
for _name in ("pymc", "pymc.sampling", "pymc.smc"):
    logging.getLogger(_name).setLevel(logging.ERROR)

SEED = 20260623

# Posterior sampling sizes. The models are tiny (F plus a few simplexes), so these
# are cheap; SMC marginal-likelihood estimates use their own draws.
DRAWS = 2000
TUNE = 2000
CHAINS = 4
SMC_DRAWS = 2000


def _prep(counts):
    """Drop empty demes and return an integer (S x K) counts matrix + row sums."""
    counts = np.asarray(counts, float)
    counts = counts[counts.sum(1) > 0]
    counts = np.rint(counts).astype(int)
    return counts, counts.sum(1)


@contextlib.contextmanager
def _quiet():
    """Silence PyMC's sampling chatter so the run-script output stays readable.

    Flush first: SMC may fork workers that would otherwise inherit and replay the
    parent's buffered stdout on exit (duplicated lines when piped to a file).
    """
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        yield


def _final_logml(idata):
    """Per-chain full-data log marginal likelihood from an SMC InferenceData.

    sample_stats['log_marginal_likelihood'] is a (chain, draw) array whose entries
    are per-stage cumulative log-ML sequences, NaN-padded to a common length and
    with a chain-dependent number of stages. The last non-NaN value of each entry
    is that chain's full-data estimate.
    """
    arr = np.asarray(idata.sample_stats["log_marginal_likelihood"].values).ravel()
    vals = []
    for entry in arr:
        seq = np.asarray(entry, dtype=float).ravel()
        seq = seq[~np.isnan(seq)]
        if seq.size:
            vals.append(seq[-1])
    return np.asarray(vals, dtype=float)


# --------------------------------------------------------------------------- #
#  Posterior for F_ST (the credible-interval replacement for the point + CI)   #
# --------------------------------------------------------------------------- #
def _bn_likelihood(count_list, f_prior=("uniform",)):
    """Build the structured (M1) Balding-Nichols model over a list of loci.

    count_list : list of (S_l x K_l) counts matrices (one per locus). A single
                 locus is just a one-element list.
    f_prior    : ('uniform',) or ('beta', a, b) -- prior on F, for sensitivity.
    Returns the pm.Model (F is the shared cultural F_ST).
    """
    model = pm.Model()
    with model:
        if f_prior[0] == "uniform":
            F = pm.Uniform("F", 0.0, 1.0)
        elif f_prior[0] == "beta":
            F = pm.Beta("F", alpha=f_prior[1], beta=f_prior[2])
        else:
            raise ValueError(f"unknown F prior {f_prior!r}")
        alpha = (1.0 - F) / F
        for j, c in enumerate(count_list):
            c, n_i = _prep(c)
            K = c.shape[1]
            pi = pm.Dirichlet(f"pi_{j}", a=np.ones(K))
            pm.DirichletMultinomial(f"x_{j}", n=n_i, a=alpha * pi, observed=c)
    return model


def _panmixia_likelihood(count_list):
    """Build the panmixia (M0) model: all demes share one pi per locus (F -> 0)."""
    model = pm.Model()
    with model:
        for j, c in enumerate(count_list):
            c, n_i = _prep(c)
            K = c.shape[1]
            pi = pm.Dirichlet(f"pi_{j}", a=np.ones(K))
            pm.Multinomial(f"x_{j}", n=n_i, p=pi, observed=c)
    return model


def fst_posterior(counts, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED,
                  f_prior=("uniform",)):
    """Posterior for single-locus cultural F_ST. Returns an ArviZ InferenceData."""
    return fst_posterior_multilocus([counts], draws, tune, chains, seed, f_prior)


def fst_posterior_multilocus(count_list, draws=DRAWS, tune=TUNE, chains=CHAINS,
                             seed=SEED, f_prior=("uniform",)):
    """Posterior for multilocus cultural F_ST (one shared F across loci)."""
    model = _bn_likelihood(count_list, f_prior=f_prior)
    with model, _quiet():
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          random_seed=seed, progressbar=False,
                          compute_convergence_checks=False)
    return idata


def fst_summary(idata, hdi_prob=0.95):
    """Posterior median, 95% HDI, mean, R-hat, ESS for F (the cultural F_ST)."""
    f = idata.posterior["F"].values.ravel()
    hdi = az.hdi(idata, var_names=["F"], hdi_prob=hdi_prob)["F"].values
    ess = float(az.ess(idata, var_names=["F"])["F"].values)
    rhat = float(az.rhat(idata, var_names=["F"])["F"].values)
    return {
        "median": float(np.median(f)),
        "mean": float(np.mean(f)),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
        "rhat": rhat,
        "ess": ess,
    }


# --------------------------------------------------------------------------- #
#  Gini-Simpson / Nei G_ST readout: a credible interval on the SAME estimator  #
#  popgen.gst reports as a point value (the mls-emergence-symmetric readout)   #
# --------------------------------------------------------------------------- #
def _hdi_bounds(samples, hdi_prob=0.95):
    """95% HDI of a 1-D sample array, robust across arviz keyword conventions."""
    x = np.asarray(samples, float).ravel()
    for kw in ("hdi_prob", "prob"):
        try:
            h = az.hdi(x, **{kw: hdi_prob})
            return float(h[0]), float(h[1])
        except TypeError:
            continue
    q = [100 * (1 - hdi_prob) / 2, 100 * (1 + hdi_prob) / 2]
    lo, hi = np.percentile(x, q)
    return float(lo), float(hi)


def gst_posterior(idata, counts_or_list, seed=0):
    """Posterior of Nei's (multilocus) G_ST on Gini-Simpson diversity for the
    OBSERVED demes, reconstructed conjugately from the Balding-Nichols fit.

    With the DirichletMultinomial(n_i, alpha*pi_l) likelihood and latent
    p_{l,i} ~ Dirichlet(alpha*pi_l), the posterior of an observed deme's
    frequency is p_{l,i} | x_{l,i} ~ Dirichlet(alpha*pi_l + x_{l,i}). For each
    posterior draw of (F, pi) we reconstruct every deme's frequencies and compute
    Nei's multilocus G_ST = (sum_l H_T,l - sum_l H_S,l) / sum_l H_T,l on
    Gini-Simpson diversity H = 1 - sum p^2. This turns the frequentist point
    estimate popgen.gst reports into a credible interval, from the same fit that
    yields the Balding-Nichols F parameter -- the readout used in ../mls-emergence.
    """
    count_list = ([counts_or_list] if np.asarray(counts_or_list[0]).ndim == 1
                  else list(counts_or_list))
    F = np.asarray(idata.posterior["F"].values).ravel()      # (S,)
    alpha = (1.0 - F) / F
    rng = np.random.default_rng(seed)
    HT = np.zeros_like(F)
    HS = np.zeros_like(F)
    for j, c in enumerate(count_list):
        c, n_i = _prep(c)
        K = c.shape[1]
        w = n_i / n_i.sum()
        pi = np.asarray(idata.posterior[f"pi_{j}"].values).reshape(-1, K)   # (S,K)
        conc = alpha[:, None, None] * pi[:, None, :] + c[None, :, :]        # (S,D,K)
        g = rng.standard_gamma(conc)
        p = g / g.sum(axis=2, keepdims=True)                               # (S,D,K)
        h_within = 1.0 - np.sum(p ** 2, axis=2)                            # (S,D)
        HS = HS + np.sum(h_within * w[None, :], axis=1)                    # (S,)
        p_pool = np.sum(p * w[None, :, None], axis=1)                      # (S,K)
        HT = HT + (1.0 - np.sum(p_pool ** 2, axis=1))                      # (S,)
    return np.where(HT > 0, (HT - HS) / HT, 0.0)


def gst_summary(idata, counts_or_list, seed=0, hdi_prob=0.95):
    """Posterior median, mean, and 95% HDI of the Gini-Simpson G_ST readout.

    The frequentist plug-in (popgen.gst / gst_multilocus) is left to the caller,
    which already computes it, so bayes.py stays decoupled from popgen.
    """
    g = gst_posterior(idata, counts_or_list, seed=seed)
    lo, hi = _hdi_bounds(g, hdi_prob)
    return {
        "gst_median": float(np.median(g)),
        "gst_mean": float(np.mean(g)),
        "gst_hdi_lo": lo,
        "gst_hdi_hi": hi,
        "gst_samples": g,
    }


# --------------------------------------------------------------------------- #
#  Bayes factor: structure (M1) vs panmixia (M0) -- the permutation-p analogue #
# --------------------------------------------------------------------------- #
def _smc_logml(count_list, structured, draws=SMC_DRAWS, chains=4, seed=SEED,
               f_prior=("uniform",)):
    """Per-chain log marginal likelihood from SMC for M1 (structured) or M0."""
    model = (_bn_likelihood(count_list, f_prior=f_prior) if structured
             else _panmixia_likelihood(count_list))
    with model, _quiet():
        # cores=1: run SMC chains sequentially. Forked workers would inherit the
        # parent's buffered report file and re-emit it on exit (duplicated output).
        idata = pm.sample_smc(draws=draws, chains=chains, cores=1,
                              random_seed=seed, progressbar=False)
    return _final_logml(idata)


def _kass_raftery(two_ln_bf):
    """Kass & Raftery (1995) evidence label for 2 ln BF10 (structure vs panmixia)."""
    a = abs(two_ln_bf)
    band = ("not worth more than a bare mention" if a < 2 else
            "positive" if a < 6 else
            "strong" if a < 10 else
            "very strong")
    direction = "for structure" if two_ln_bf > 0 else "for panmixia"
    return f"{band} {direction}"


def bayes_factor_structure(counts_or_list, draws=SMC_DRAWS, chains=4, seed=SEED,
                           f_prior=("uniform",)):
    """Bayes factor for between-deme structure vs panmixia.

    Accepts one counts matrix or a list of per-locus matrices. Fits M1 and M0 with
    SMC, returns a dict with per-chain and combined log marginal likelihoods,
    log10 BF10, 2 ln BF10, the Kass-Raftery label, and the across-chain spread of
    2 ln BF10 as a stability check. f_prior sets the prior on F in M1 (M0 has no F),
    for the prior-sensitivity check.
    """
    count_list = ([counts_or_list] if np.asarray(counts_or_list[0]).ndim == 1
                  else list(counts_or_list))
    lml1 = _smc_logml(count_list, structured=True, draws=draws, chains=chains,
                      seed=seed, f_prior=f_prior)
    lml0 = _smc_logml(count_list, structured=False, draws=draws, chains=chains, seed=seed)
    m1, m0 = float(lml1.mean()), float(lml0.mean())
    ln_bf = m1 - m0
    two_ln_bf = 2.0 * ln_bf
    # across-chain spread of the two log-ML estimates, as a stability check
    chain_sd = float(2.0 * np.sqrt(lml1.var() / len(lml1) + lml0.var() / len(lml0)))
    return {
        "logml_structure": m1,
        "logml_panmixia": m0,
        "log10_bf": float(ln_bf / np.log(10)),
        "two_ln_bf": two_ln_bf,
        "two_ln_bf_chain_sd": chain_sd,
        "evidence": _kass_raftery(two_ln_bf),
    }


# --------------------------------------------------------------------------- #
#  Isolation by distance: Bayesian distance-decay (the Mantel analogue)        #
# --------------------------------------------------------------------------- #
def _triu(D):
    return np.asarray(D)[np.triu_indices_from(np.asarray(D), k=1)]


def ibd_regression(Dcomp, Dgeo, region=None, draws=DRAWS, tune=TUNE,
                   chains=CHAINS, seed=SEED):
    """Bayesian isolation-by-distance: regress pairwise compositional distance on
    geographic distance (and optionally a same/different-region indicator).

    Both distance vectors are standardized, so the geographic slope is on a
    correlation-like scale directly comparable to the Mantel r it replaces; with a
    region term it is the partial-Mantel analogue (distance effect controlling for
    the coarse regional contrast). CAVEAT: pairwise distances are not independent
    (each deme appears in many pairs), so the posterior SD understates uncertainty.
    As in the frequentist version, IBD is a secondary check, not load-bearing.

    Returns a dict: slope median + 95% HDI + P(slope > 0), and the region slope if
    a region vector was supplied.
    """
    y = _triu(Dcomp)
    x = _triu(Dgeo)
    ys = (y - y.mean()) / y.std()
    xs = (x - x.mean()) / x.std()
    with pm.Model(), _quiet():
        b0 = pm.Normal("b0", 0.0, 1.0)
        b_geo = pm.Normal("b_geo", 0.0, 1.0)
        mu = b0 + b_geo * xs
        if region is not None:
            z = _triu(region)
            zs = (z - z.mean()) / (z.std() if z.std() > 0 else 1.0)
            b_reg = pm.Normal("b_reg", 0.0, 1.0)
            mu = mu + b_reg * zs
        sigma = pm.HalfNormal("sigma", 1.0)
        pm.Normal("y", mu=mu, sigma=sigma, observed=ys)
        idata = pm.sample(draws, tune=tune, chains=chains, cores=1,
                          random_seed=seed, progressbar=False,
                          compute_convergence_checks=False)
    slope = idata.posterior["b_geo"].values.ravel()
    hdi = az.hdi(idata, var_names=["b_geo"], hdi_prob=0.95)["b_geo"].values
    out = {
        "slope_median": float(np.median(slope)),
        "slope_hdi_lo": float(hdi[0]),
        "slope_hdi_hi": float(hdi[1]),
        "p_positive": float((slope > 0).mean()),
        "controlled": region is not None,
    }
    return out


__all__ = [
    "fst_posterior", "fst_posterior_multilocus", "fst_summary",
    "gst_posterior", "gst_summary",
    "bayes_factor_structure", "ibd_regression",
]
