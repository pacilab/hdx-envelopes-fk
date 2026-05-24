"""
hdx_backexchange — Realistic HDX envelopes with heterogeneous back exchange.

Reference
---------
A. Grimaldi, E. Paci (2026), 
    Appendix F: "Heterogeneous back exchange at pattern level".

A site i that is deuterated at the end of labeling retains its deuteron with
probability β_i (and loses it with probability 1 - β_i), independently across
sites. At the level of the multivariate generating function, this corresponds
to the substitution

    z_i  →  1 - β_i + β_i z_i ,

which, when applied to Eq. C5 of the manuscript, gives the post-back-exchange
uptake generating function (z_i = z)

    H*_P(z, t) = Σ_{A⊆P} ψ_A(t) · Π_{i∈A} β_i(1 - z) · Π_{j∈P\A} (1 - β_j + β_j z) .

The coefficients of z^m yield the distribution of the number of *retained*
deuterons m_P after back exchange. The centroid simplifies to

    E[m_P(t)] = Σ_{i∈P} β_i · (1 - ψ_i(t)) ,

which depends on single-residue survivals only.

The module is independent of the regime used to compute the subset survivals
{ψ_A(t)}: any of the four classes in :mod:`hdx_env` (UncoupledResidues,
ConcertedSwitching, MetastableBasins, FullMarkov) can be passed in, since they
all share the ``all_subset_survivals`` API.

Public API
----------
    uptake_with_backexchange(P, t, psi_X_dict, beta)
        Post-back-exchange distribution of retained deuterons (eq. F).
    centroid_with_backexchange(P, t, psi_X_dict, beta)
        Mean retained deuterons (closed form, single-site only).
    variance_with_backexchange(P, t, psi_X_dict, beta)
        Variance of retained deuterons (uses pair survivals).
    envelope_with_backexchange(model, P, t, beta)
        High-level wrapper: takes a model exposing ``all_subset_survivals``.
    mass_envelope_with_backexchange(model, P, t, beta, nat_env, n_peaks=None)
        Convolve the retained-deuteron distribution with a natural isotope
        envelope.

Random-rate generators (for EX2 demonstrations)
-----------------------------------------------
    random_pfactors(N, log10_range=(0.0, 6.0), rng=None)
    random_kint(N, log10_range=(-2.0, 1.0), rng=None)
    random_betas(N, low=0.5, high=0.9, rng=None)
    ex2_uncoupled(pfact, k_int, k_total=1e4, rng=None)
        Build an :class:`UncoupledResidues` in the EX2 (fast-switch) regime
        whose per-site equilibrium open probability is π_i = 1/(1 + P_i).
"""

from __future__ import annotations

import numpy as np

from hdx_env import (
    UncoupledResidues,
    uptake_from_survivals,
    moments_from_distribution,
    _all_subsets,
)


# =====================================================================
# Core back-exchange transformation (Appendix F)
# =====================================================================

def _poly_one_minus_z_power(k):
    """Coefficients of (1 - z)^k, length k+1, in ascending powers of z."""
    coef = np.array([1.0])
    step = np.array([1.0, -1.0])
    for _ in range(k):
        coef = np.convolve(coef, step)
    return coef


def _poly_retention(beta_vec):
    """Coefficients of Π_j (1 - β_j + β_j z) in ascending powers of z."""
    coef = np.array([1.0])
    for b in beta_vec:
        coef = np.convolve(coef, np.array([1.0 - b, b]))
    return coef


def uptake_with_backexchange(P, t, psi_X_dict, beta):
    """Post-back-exchange uptake distribution from a family of subset survivals.

    Implements Appendix F by expanding H*_P(z, t) as a polynomial in z and
    summing contributions over all subsets A ⊆ P.

    Parameters
    ----------
    P : iterable of site indices
        Peptide residues (positions in the model). Sorted internally.
    t : array_like of times
        Used only to infer the time-axis length from ``psi_X_dict``.
    psi_X_dict : dict {tuple(X): array of shape (nT,)}
        Subset survivals for all X ⊆ P (the empty tuple is inserted as
        ψ_∅ ≡ 1 if missing).
    beta : array_like of shape (|P|,)
        Per-site retention probabilities β_i ∈ [0, 1], ordered by the sorted
        order of P.

    Returns
    -------
    prob : ndarray of shape (nT, |P| + 1)
        prob[ti, m] = P(m_P(t_i) = m) after back exchange.
    """
    P = tuple(sorted(P))
    nP = len(P)
    beta = np.asarray(beta, dtype=float)
    if beta.shape != (nP,):
        raise ValueError(f"beta must have shape ({nP},), got {beta.shape}")
    if np.any(beta < 0) or np.any(beta > 1):
        raise ValueError("beta entries must lie in [0, 1]")

    sample = np.asarray(next(iter(psi_X_dict.values())), dtype=float)
    nT = sample.shape[0]
    if () not in psi_X_dict:
        psi_X_dict = dict(psi_X_dict)
        psi_X_dict[()] = np.ones(nT)

    pos = {i: k for k, i in enumerate(P)}
    out = np.zeros((nT, nP + 1))

    for X, psi in psi_X_dict.items():
        psi_arr = np.asarray(psi, dtype=float)
        idx_X = [pos[i] for i in X]
        idx_notX = [pos[i] for i in P if i not in set(X)]
        B_X = float(np.prod(beta[idx_X])) if idx_X else 1.0
        poly_X = _poly_one_minus_z_power(len(X))                # length |X|+1
        poly_notX = _poly_retention(beta[idx_notX])             # length |P\X|+1
        coef = np.convolve(poly_X, poly_notX) * B_X             # length |P|+1
        out += psi_arr[:, None] * coef[None, :]

    out = np.clip(out, 0.0, None)
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def centroid_with_backexchange(P, t, psi_X_dict, beta):
    """Closed-form mean retained deuterons,  E[m_P(t)] = Σ_i β_i (1 - ψ_i(t)).
    Only single-residue survivals are needed.
    """
    P = tuple(sorted(P))
    beta = np.asarray(beta, dtype=float)
    sample = np.asarray(next(iter(psi_X_dict.values())), dtype=float)
    nT = sample.shape[0]
    out = np.zeros(nT)
    for k, i in enumerate(P):
        psi_i = np.asarray(psi_X_dict[(i,)], dtype=float)
        out = out + beta[k] * (1.0 - psi_i)
    return out


def variance_with_backexchange(P, t, psi_X_dict, beta):
    """Variance of m_P(t) under heterogeneous back exchange.

    Using m_i = b_i n_i with b_i Bernoulli(β_i) independent across i,

        Var[m_P] = Σ_i β_i (1 - β_i ψ_i)(1 - (1 - β_i)(1 - ψ_i)) ... (not used)

    The clean expression in terms of survivals is

        E[m_i]      = β_i (1 - ψ_i)
        E[m_i m_j]  = β_i β_j (1 - ψ_i - ψ_j + ψ_ij)        (i ≠ j)
        Var[m_P]    = Σ_i β_i(1-ψ_i) − Σ_i β_i^2 (1-ψ_i)^2
                      + 2 Σ_{i<j} β_i β_j [(1 - ψ_i - ψ_j + ψ_ij) − (1-ψ_i)(1-ψ_j)] .
    """
    P = tuple(sorted(P))
    beta = np.asarray(beta, dtype=float)
    sample = np.asarray(next(iter(psi_X_dict.values())), dtype=float)
    nT = sample.shape[0]
    psi_i = np.array([np.asarray(psi_X_dict[(i,)], dtype=float) for i in P])  # (nP, nT)
    p_i = 1.0 - psi_i                                                          # P(n_i=1)
    var = (beta[:, None] * p_i - (beta[:, None] ** 2) * (p_i ** 2)).sum(axis=0)
    nP = len(P)
    for a in range(nP):
        for b in range(a + 1, nP):
            i, j = P[a], P[b]
            psi_ij = np.asarray(psi_X_dict[tuple(sorted((i, j)))], dtype=float)
            cov_n = (1.0 - psi_i[a] - psi_i[b] + psi_ij) - p_i[a] * p_i[b]
            var = var + 2.0 * beta[a] * beta[b] * cov_n
    return var


# =====================================================================
# High-level wrappers
# =====================================================================

def envelope_with_backexchange(model, P, t, beta):
    """One-shot helper.

    Parameters
    ----------
    model : object exposing ``all_subset_survivals(P, t) -> dict``
        Any of the four regime classes in :mod:`hdx_env`.
    P : iterable of site indices
    t : 1D array of labeling times
    beta : array of retention probabilities, len(P)

    Returns
    -------
    prob_with_bx : ndarray (nT, |P|+1)
        Post-back-exchange uptake distribution.
    prob_no_bx : ndarray (nT, |P|+1)
        Pre-back-exchange distribution (β_i ≡ 1), returned for comparison.
    psi_X_dict : dict
        The subset survivals used (re-usable for downstream observables).
    """
    P = tuple(sorted(P))
    psi_X_dict = model.all_subset_survivals(P, t)
    prob_no_bx = uptake_from_survivals(P, psi_X_dict)
    prob_with_bx = uptake_with_backexchange(P, t, psi_X_dict, beta)
    return prob_with_bx, prob_no_bx, psi_X_dict


def mass_envelope_with_backexchange(uptake_dist, nat_env, n_peaks=None):
    """Convolve a (post- or pre-back-exchange) uptake distribution with the
    natural isotope envelope. Mirrors :func:`hdx_env.mass_envelope` for
    convenience; kept here so that downstream notebooks need to import only
    this module.
    """
    uptake_dist = np.asarray(uptake_dist, dtype=float)
    nat_env = np.asarray(nat_env, dtype=float)
    nT, nU1 = uptake_dist.shape
    nU = nU1 - 1
    M = len(nat_env)
    L = nU + M - 1 if n_peaks is None else n_peaks
    out = np.zeros((nT, L))
    for ti in range(nT):
        conv = np.convolve(uptake_dist[ti], nat_env)
        out[ti, :min(L, len(conv))] = conv[:L]
    return out


# =====================================================================
# Random-parameter generators for EX2 demonstrations
# =====================================================================

def random_pfactors(N, log10_range=(0.0, 6.0), rng=None):
    """Log-uniform protection factors P_i ∈ [10^a, 10^b].

    P_i = 1 corresponds to a fully exposed residue (π_open = 1/2),
    P_i ≫ 1 to highly protected residues. The default range spans the
    physically observed window for folded proteins.
    """
    if rng is None:
        rng = np.random.default_rng()
    a, b = log10_range
    return 10.0 ** rng.uniform(a, b, size=N)


def random_kint(N, log10_range=(-2.0, 1.0), rng=None):
    """Log-uniform intrinsic exchange rates (in the same time unit used for t).

    Default range mimics base-catalysed amide intrinsic rates around
    neutral pH and physiological temperature (a few decades of variation
    across residue types).
    """
    if rng is None:
        rng = np.random.default_rng()
    a, b = log10_range
    return 10.0 ** rng.uniform(a, b, size=N)


def random_betas(N, low=0.5, high=0.9, rng=None):
    """Uniformly sampled retention probabilities β_i ∈ [low, high].

    Typical HDX–MS experiments report back-exchange fractions in the
    range 10%–50%, so β_i ≈ 0.5–0.9 is a realistic default.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(low, high, size=N)


def setup_uncoupled(pfact, k_int, k_total=1e4, rng=None):
    """Construct an :class:`UncoupledResidues` model in the pre-equilibrium (fast-switch)
    regime from protection factors and intrinsic rates.

    Each site has equilibrium open probability π_i = 1/(1 + P_i). The
    closing/opening rates are set to

        k_op_i = k_total · π_i ,
        k_cl_i = k_total · (1 - π_i) ,

    so that k_op_i + k_cl_i = k_total is large compared to k_int_i and the
    site exchanges with effective rate k_eff_i = π_i · k_int_i.

    Parameters
    ----------
    pfact : array_like, shape (N,)
        Protection factors P_i.
    k_int : array_like, shape (N,)
        Intrinsic exchange rates.
    k_total : float
        Total per-site interconversion rate (must satisfy k_total ≫ k_int_i
        for the pre-equilibrium approximation limit to apply). Default 1e4.

    Returns
    -------
    model : :class:`UncoupledResidues`
    """
    pfact = np.asarray(pfact, dtype=float)
    k_int = np.asarray(k_int, dtype=float)
    if pfact.shape != k_int.shape:
        raise ValueError("pfact and k_int must have the same shape")
    pi_open = 1.0 / (1.0 + pfact)
    k_op = k_total * pi_open
    k_cl = k_total * (1.0 - pi_open)
    return UncoupledResidues(k_op, k_cl, k_int)


__all__ = [
    "uptake_with_backexchange",
    "centroid_with_backexchange",
    "variance_with_backexchange",
    "envelope_with_backexchange",
    "mass_envelope_with_backexchange",
    "random_pfactors",
    "random_kint",
    "random_betas",
    "setup_uncoupled",
]
