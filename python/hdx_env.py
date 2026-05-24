"""
hdx_env — Simulation of HDX conformational dynamics and uptake envelopes.

Reference
---------
A. Grimaldi, E. Paci (2026).
    How conformational dynamics shape hydrogen--deuterium exchange
    isotopic envelopes.

Four reduced descriptions of conformational dynamics are implemented, all of
which map onto the same peptide-level observable (uptake distribution, isotopic
envelope) via the family of subset survival probabilities {ψ_X(t)}:

    UncoupledResidues       — each amide an independent two-state Linderstrom-Lang
    ConcertedSwitching      — one shared binary switch controls a subset of amides
    MetastableBasins        — metastable basins with slow (optionally frozen)
                                interbasin transitions; intrabasin fast equilibrium
    FullMarkov              — general continuous-time Markov chain on S = {0,1}^N

Each regime exposes a common API
    .psi_X(X, t)                    -> ψ_X(t),   shape (nT,)
    .all_subset_survivals(P, t)     -> dict {tuple(X): array(nT)}
    .uptake_distribution(P, t)      -> shape (nT, |P|+1)
    .centroid(P, t), .variance(P, t)

Conventions
-----------
- All rates are expressed with the same time unit (the user picks a global
  timescale by choosing numerical values of rates; time `t` is then in those
  same units).
- Site indices are 0..N-1.
- For the Full regime, the state index s is a bitmask with bit i = s_i.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from math import comb
from scipy.linalg import null_space

# =====================================================================
# Core helpers
# =====================================================================

def _all_subsets(P):
    """Iterate over all subsets of P (returned as sorted tuples, including the empty set)."""
    P = tuple(sorted(P))
    for k in range(len(P) + 1):
        for combo in combinations(P, k):
            yield combo


def uptake_from_survivals(P, psi_X):
    """Probability distribution of the peptide uptake n_P(t) from the family
    {ψ_X(t)}_{X ⊆ P}, using eq. (C10):

        P(n_P = n, t) = Σ_{X⊆P, |X|≥|P|-n} (-1)^(|X|-|P|+n) C(|X|,|P|-n) ψ_X(t)

    Parameters
    ----------
    P : iterable of site indices
    psi_X : dict whose keys are tuples X (sorted) and values arrays ψ_X(t)
        (any iterable-of-the-same-length works). The empty tuple () must map to
        ψ_∅(t) ≡ 1 (if missing, it is inserted).

    Returns
    -------
    prob : ndarray of shape (nT, |P|+1)
        prob[t, n] = P(n_P(t) = n).
    """
    P = tuple(sorted(P))
    nP = len(P)
    # Infer nT and make sure empty set is present
    sample_val = next(iter(psi_X.values()))
    sample_arr = np.asarray(sample_val, dtype=float)
    nT = sample_arr.shape[0]
    if () not in psi_X:
        psi_X = dict(psi_X)
        psi_X[()] = np.ones(nT)
    prob = np.zeros((nT, nP + 1))
    for X, psi in psi_X.items():
        sizeX = len(X)
        psi_arr = np.asarray(psi, dtype=float)
        for n in range(nP + 1):
            if sizeX < nP - n:
                continue
            sign = (-1) ** (sizeX - nP + n)
            prob[:, n] += sign * comb(sizeX, nP - n) * psi_arr
    # Numerical hygiene: clip tiny negatives from round-off
    prob = np.clip(prob, 0.0, None)
    # Renormalise so that rows sum to 1 (guard against O(ε) drift)
    s = prob.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    prob = prob / s
    return prob


def moments_from_distribution(prob):
    """Mean, variance of the uptake from its distribution prob[t, n]."""
    nU = prob.shape[1] - 1
    k = np.arange(nU + 1)
    m1 = (prob * k).sum(axis=1)
    m2 = (prob * k ** 2).sum(axis=1)
    return m1, m2 - m1 ** 2


# =====================================================================
# Single-residue Linderstrom--Lang survival (used by several regimes)
# =====================================================================

def _ll_survival_conditional(kop, kcl, kint, t):
    """
    Conditional survivals ψ(0,t), ψ(1,t) for a two-state LL model with
    killing rate k_int in the open state.

    Numerically stable decomposition onto the two non-positive eigenmodes
    λ± = -(γ ∓ Δ)/2, with γ = k_op + k_cl + k_int and Δ = sqrt(γ² - 4 k_op k_int).

        ψ(0,t) = ½(1+γ/Δ) e^(λ_- t) + ½(1-γ/Δ) e^(λ_+ t)
        ψ(1,t) = ½(1+(γ-2 k_int)/Δ) e^(λ_- t) + ½(1-(γ-2 k_int)/Δ) e^(λ_+ t)

    The Δ → 0 (double-root) limit is handled separately via the Taylor form.
    """
    t = np.asarray(t, dtype=float)
    gamma = kop + kcl + kint
    disc = gamma ** 2 - 4.0 * kop * kint
    if disc < 0:
        disc = 0.0                                    # theoretical guarantee
    Delta = np.sqrt(disc)
    if Delta > 1e-12 * max(gamma, 1.0):
        lam_m = -(gamma - Delta) / 2.0                # less negative (slow)
        lam_p = -(gamma + Delta) / 2.0                # more negative (fast)
        em = np.exp(lam_m * t)
        ep = np.exp(lam_p * t)
        a_m0 = 0.5 * (1.0 + gamma / Delta)
        a_p0 = 0.5 * (1.0 - gamma / Delta)
        a_m1 = 0.5 * (1.0 + (gamma - 2.0 * kint) / Delta)
        a_p1 = 0.5 * (1.0 - (gamma - 2.0 * kint) / Delta)
        psi0 = a_m0 * em + a_p0 * ep
        psi1 = a_m1 * em + a_p1 * ep
    else:                                             # Δ ≈ 0 : Taylor limit
        pref = np.exp(-gamma * t / 2.0)
        psi0 = pref * (1.0 + gamma * t / 2.0)
        psi1 = pref * (1.0 + (gamma - 2.0 * kint) * t / 2.0)
    return psi0, psi1


def _ll_survival(kop, kcl, kint, t, p0=None):
    """
    Survival ψ(t) averaged over initial condition. p0 = (P(s=0), P(s=1));
    default is equilibrium π_cl = kcl/(kop+kcl), π_op = kop/(kop+kcl).
    """
    if p0 is None:
        denom = kop + kcl
        if denom <= 0:
            p0 = (1.0, 0.0)
        else:
            p0 = (kcl / denom, kop / denom)
    psi0, psi1 = _ll_survival_conditional(kop, kcl, kint, t)
    return p0[0] * psi0 + p0[1] * psi1


# =====================================================================
# Regime 1 — Uncoupled residues
# =====================================================================

class UncoupledResidues:
    """
    N independent Linderstrom--Lang amides.

    Parameters
    ----------
    k_op, k_cl, k_int : array_like, shape (N,)
        Opening, closing, intrinsic exchange rates per site.
    p0 : array_like of shape (N, 2), optional
        Initial distribution per site: p0[i] = (P(s_i=0), P(s_i=1)).
        Defaults to the equilibrium of each site.
    """

    name = "uncoupled"

    def __init__(self, k_op, k_cl, k_int, p0=None):
        self.k_op = np.asarray(k_op, dtype=float)
        self.k_cl = np.asarray(k_cl, dtype=float)
        self.k_int = np.asarray(k_int, dtype=float)
        assert self.k_op.shape == self.k_cl.shape == self.k_int.shape
        self.N = int(self.k_op.size)
        if p0 is None:
            denom = self.k_op + self.k_cl
            denom_safe = np.where(denom > 0, denom, 1.0)
            pi_op = np.where(denom > 0, self.k_op / denom_safe, 0.0)
            p0 = np.stack([1.0 - pi_op, pi_op], axis=1)
        self.p0 = np.asarray(p0, dtype=float)
        assert self.p0.shape == (self.N, 2)

    # ---- observables ---------------------------------------------------
    def psi_i(self, i, t):
        return _ll_survival(self.k_op[i], self.k_cl[i], self.k_int[i], t,
                            p0=self.p0[i])

    def psi_X(self, X, t):
        t = np.asarray(t, dtype=float)
        out = np.ones_like(t)
        for i in X:
            out = out * self.psi_i(i, t)
        return out

    def all_subset_survivals(self, P, t):
        P = tuple(sorted(P))
        psi_site = {i: self.psi_i(i, t) for i in P}
        subs = {}
        for X in _all_subsets(P):
            arr = np.ones(len(np.atleast_1d(t)))
            for i in X:
                arr = arr * psi_site[i]
            subs[X] = arr
        return subs

    def uptake_distribution(self, P, t):
        """
        uncoupled residues : uptake is a sum of independent Bernoullis with success 1 - ψ_i(t).
        """
        P = tuple(sorted(P))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        nT = len(t)
        dist = np.ones((nT, 1))
        for i in P:
            psi_i = self.psi_i(i, t)
            new_dist = np.zeros((nT, dist.shape[1] + 1))
            new_dist[:, :-1] += dist * psi_i[:, None]
            new_dist[:, 1:] += dist * (1.0 - psi_i)[:, None]
            dist = new_dist
        return dist

    def centroid(self, P, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        c = np.zeros_like(t)
        for i in P:
            c = c + (1.0 - self.psi_i(i, t))
        return c

    def variance(self, P, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        v = np.zeros_like(t)
        for i in P:
            psi = self.psi_i(i, t)
            v = v + psi * (1.0 - psi)
        return v


# =====================================================================
# Regime 2 — Concerted switching
# =====================================================================

class ConcertedSwitching:
    """
    All N residues share a single binary switch q ∈ {0,1} with rates
    (k_op_q, k_cl_q). When q = 0 the whole set is closed; when q = 1, open.
    Intrinsic exchange rates k_int may differ across residues.
    """

    name = "concerted"

    def __init__(self, k_op_q, k_cl_q, k_int, p0_q=None):
        self.k_op_q = float(k_op_q)
        self.k_cl_q = float(k_cl_q)
        self.k_int = np.asarray(k_int, dtype=float)
        self.N = int(self.k_int.size)
        if p0_q is None:
            denom = self.k_op_q + self.k_cl_q
            if denom > 0:
                pi = self.k_op_q / denom
            else:
                pi = 0.0
            p0_q = (1.0 - pi, pi)
        self.p0_q = tuple(float(x) for x in p0_q)

    # ---- observables ---------------------------------------------------
    def psi_X(self, X, t):
        """
        Uses the scalar map eq. (E11): kint_X = Σ_{i∈X} k_int_i.
        """
        X = tuple(sorted(X))
        if len(X) == 0:
            return np.ones_like(np.atleast_1d(np.asarray(t, dtype=float)))
        kint_X = float(np.sum(self.k_int[list(X)]))
        return _ll_survival(self.k_op_q, self.k_cl_q, kint_X, t, p0=self.p0_q)

    def all_subset_survivals(self, P, t):
        P = tuple(sorted(P))
        return {X: self.psi_X(X, t) for X in _all_subsets(P)}

    def uptake_distribution(self, P, t):
        return uptake_from_survivals(P, self.all_subset_survivals(P, t))

    def centroid(self, P, t):
        prob = self.uptake_distribution(P, t)
        m, _ = moments_from_distribution(prob)
        return m

    def variance(self, P, t):
        prob = self.uptake_distribution(P, t)
        _, v = moments_from_distribution(prob)
        return v


# =====================================================================
# Regime 3 — Metastable basins
# =====================================================================

class MetastableBasins:
    """Metastable basin description. For each of B basins and each of N sites,
    the basin-resolved open fraction π^(b)_i is provided. The coarse-grained
    dynamics is a Markov chain on basin labels with optional rate matrix
    W_basin (B, B). If W_basin is None, basins are frozen (no interbasin
    transitions during the labelling interval, eq. E23).
    """

    name = "basins"

    def __init__(self, pi_open, k_int, p0_basin=None, W_basin=None):
        self.pi_open = np.asarray(pi_open, dtype=float)
        assert self.pi_open.ndim == 2
        self.B, self.N = self.pi_open.shape
        self.k_int = np.asarray(k_int, dtype=float)
        assert self.k_int.shape == (self.N,)
        self.W_basin = None if W_basin is None else np.asarray(W_basin, dtype=float)
        if self.W_basin is not None:
            assert self.W_basin.shape == (self.B, self.B)
            Wz = self.W_basin.copy()
            np.fill_diagonal(Wz, 0.0)
            self.L_basin = Wz - np.diag(Wz.sum(axis=1))
        else:
            self.L_basin = None
        if p0_basin is None:
            if self.L_basin is not None:
                ns = null_space(self.L_basin.T)
                if ns.shape[1] >= 1:
                    pi = np.real(ns[:, 0])
                    pi = pi / pi.sum()
                    p0_basin = pi
                else:
                    p0_basin = np.ones(self.B) / self.B
            else:
                p0_basin = np.ones(self.B) / self.B
        self.p0_basin = np.asarray(p0_basin, dtype=float)

    # ---- observables ---------------------------------------------------
    def _r_X(self, X):
        X = list(X)
        if len(X) == 0:
            return np.zeros(self.B)
        return (self.k_int[X] * self.pi_open[:, X]).sum(axis=1)  # shape (B,)

    def psi_X(self, X, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        r = self._r_X(X)
        if self.L_basin is None:
            # Frozen basins: ψ_X(b, t) = exp(-r(b) t)
            psi_b = np.exp(-np.outer(t, r))  # (nT, B)
            return psi_b @ self.p0_basin
        else:
            # Feynman-Kac on basin chain: ∂t ψ = (L - diag(r)) ψ, ψ(b,0)=1
            A = self.L_basin - np.diag(r)
            w, V = np.linalg.eig(A)
            Vinv = np.linalg.inv(V)
            ones = np.ones(self.B)
            c = Vinv @ ones
            exp_wt = np.exp(np.outer(t, w))  # (nT, B)
            psi_b = (exp_wt * c[None, :]) @ V.T  # (nT, B)
            psi_b = psi_b.real
            return psi_b @ self.p0_basin

    def all_subset_survivals(self, P, t):
        P = tuple(sorted(P))
        return {X: self.psi_X(X, t) for X in _all_subsets(P)}

    def uptake_distribution(self, P, t):
        P = tuple(sorted(P))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        nP = len(P)
        nT = len(t)
        if self.L_basin is None:
            # Closed-form mixture of uncoupled residues distributions (eq. E24)
            out = np.zeros((nT, nP + 1))
            for b in range(self.B):
                dist = np.ones((nT, 1))
                for i in P:
                    psi_i = np.exp(-self.k_int[i] * self.pi_open[b, i] * t)
                    new_dist = np.zeros((nT, dist.shape[1] + 1))
                    new_dist[:, :-1] += dist * psi_i[:, None]
                    new_dist[:, 1:] += dist * (1.0 - psi_i)[:, None]
                    dist = new_dist
                out += self.p0_basin[b] * dist
            return out
        else:
            return uptake_from_survivals(P, self.all_subset_survivals(P, t))

    def centroid(self, P, t):
        prob = self.uptake_distribution(P, t)
        m, _ = moments_from_distribution(prob)
        return m

    def variance(self, P, t):
        prob = self.uptake_distribution(P, t)
        _, v = moments_from_distribution(prob)
        return v


# =====================================================================
# Regime 4 — Full Markov chain on {0,1}^N
# =====================================================================

class FullMarkov:
    """General continuous-time Markov chain on the coarse-grained state space
    S = {0,1}^N. A 2^N × 2^N transition-rate matrix W is supplied (its
    diagonal is ignored), together with intrinsic exchange rates k_int per
    residue. Exact for moderate N (N ≲ 10); set `N ≲ 10` for the deterministic
    subset-survival computation, or use simulate() for larger systems.
    """

    name = "full"

    def __init__(self, W, k_int, p0=None):
        self.W = np.asarray(W, dtype=float)
        self.k_int = np.asarray(k_int, dtype=float)
        self.N = int(self.k_int.size)
        self.dim = 2 ** self.N
        assert self.W.shape == (self.dim, self.dim), \
            f"W shape {self.W.shape} != (2^N, 2^N) = ({self.dim},{self.dim})"
        Wz = self.W.copy()
        np.fill_diagonal(Wz, 0.0)
        self.L = Wz - np.diag(Wz.sum(axis=1))
        # state[s, i] = s_i (bit i of state index s)
        self.state_bits = np.array(
            [[(s >> i) & 1 for i in range(self.N)] for s in range(self.dim)],
            dtype=int,
        )
        if p0 is None:
            ns = null_space(self.L.T)
            if ns.shape[1] >= 1:
                pi = np.real(ns[:, 0])
                pi = pi / pi.sum()
                p0 = pi
            else:
                p0 = np.ones(self.dim) / self.dim
        self.p0 = np.asarray(p0, dtype=float)
        assert self.p0.shape == (self.dim,)

    # ---- observables ---------------------------------------------------
    def _r_vec(self, X):
        r = np.zeros(self.dim)
        for i in X:
            r += self.k_int[i] * self.state_bits[:, i]
        return r

    def psi_X(self, X, t, eig_cache=None):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        X = tuple(sorted(X))
        key = X
        if eig_cache is not None and key in eig_cache:
            w, V, Vinv = eig_cache[key]
        else:
            A = self.L - np.diag(self._r_vec(X))
            w, V = np.linalg.eig(A)
            Vinv = np.linalg.inv(V)
            if eig_cache is not None:
                eig_cache[key] = (w, V, Vinv)
        ones = np.ones(self.dim)
        c = Vinv @ ones  # (dim,)
        a = self.p0 @ V  # (dim,)
        coef = a * c  # (dim,)
        exp_wt = np.exp(np.outer(t, w))  # (nT, dim)
        return (exp_wt @ coef).real

    def all_subset_survivals(self, P, t):
        P = tuple(sorted(P))
        subs = {}
        cache = {}
        for X in _all_subsets(P):
            subs[X] = self.psi_X(X, t, eig_cache=cache)
        return subs

    def uptake_distribution(self, P, t):
        return uptake_from_survivals(P, self.all_subset_survivals(P, t))

    def centroid(self, P, t):
        prob = self.uptake_distribution(P, t)
        m, _ = moments_from_distribution(prob)
        return m

    def variance(self, P, t):
        prob = self.uptake_distribution(P, t)
        _, v = moments_from_distribution(prob)
        return v

    # ---- stochastic simulation -----------------------------------------
    def simulate(self, t_grid, n_traj=10_000, rng=None):
        """Gillespie/SSA simulation of the coarse-grained process with an
        additional killing (exchange) event per residue. Returns the uptake
        distribution on t_grid, shape (nT, N+1).
        """
        if rng is None:
            rng = np.random.default_rng()
        t_grid = np.asarray(t_grid, dtype=float)
        t_max = float(t_grid[-1])
        nT = len(t_grid)
        counts = np.zeros((nT, self.N + 1), dtype=np.int64)
        # Pre-compute off-diagonal rows of W
        W = self.W.copy()
        np.fill_diagonal(W, 0.0)
        row_rates = W.sum(axis=1)  # (dim,)

        for _ in range(n_traj):
            # Sample initial state
            s = int(rng.choice(self.dim, p=self.p0 / self.p0.sum()))
            exchanged = np.zeros(self.N, dtype=bool)
            t = 0.0
            idx = 0  # next time-grid index
            while True:
                # Current hazard: conformational + exchange from active sites
                conf_rate = row_rates[s]
                open_active = (self.state_bits[s] == 1) & (~exchanged)
                exch_rate_per = self.k_int * open_active
                exch_rate = exch_rate_per.sum()
                total = conf_rate + exch_rate
                if total <= 0:
                    # Absorbing — fill remaining grid
                    while idx < nT and t_grid[idx] >= t:
                        counts[idx, int(exchanged.sum())] += 1
                        idx += 1
                    break
                dt = rng.exponential(1.0 / total)
                t_new = t + dt
                # Record all grid points crossed in [t, t_new)
                while idx < nT and t_grid[idx] < t_new:
                    counts[idx, int(exchanged.sum())] += 1
                    idx += 1
                if idx >= nT or t_new > t_max:
                    break
                # Draw event
                u = rng.random() * total
                if u < conf_rate:
                    # Conformational transition from s
                    probs = W[s] / conf_rate
                    s = int(rng.choice(self.dim, p=probs))
                else:
                    # Exchange event on an active site
                    u2 = u - conf_rate
                    cum = np.cumsum(exch_rate_per)
                    i = int(np.searchsorted(cum, u2, side='right'))
                    exchanged[i] = True
                t = t_new
        return counts / n_traj


# =====================================================================
# Random rate-matrix generators (for the Full regime)
# =====================================================================

def random_singleflip_W(N, rng=None, scale=1.0, distribution='exponential'):
    """Random (2^N × 2^N) rate matrix with transitions restricted to single
    bit flips (physically: one residue opens/closes per elementary event).
    """
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** N
    W = np.zeros((dim, dim))
    for s in range(dim):
        for i in range(N):
            sp = s ^ (1 << i)
            if distribution == 'exponential':
                W[s, sp] = rng.exponential(scale)
            elif distribution == 'uniform':
                W[s, sp] = rng.uniform(0, scale)
            else:
                raise ValueError(distribution)
    return W


def random_dense_W(N, rng=None, scale=1.0):
    """Fully dense random rate matrix (every off-diagonal has an exponential
    rate). Not restricted to single flips."""
    if rng is None:
        rng = np.random.default_rng()
    dim = 2 ** N
    W = rng.exponential(scale, size=(dim, dim))
    np.fill_diagonal(W, 0.0)
    return W


# =====================================================================
# Natural isotopic envelope (averagine model) + mass envelope
# =====================================================================

# Monoisotopic masses and natural abundances (NIST / IUPAC)
_ISOTOPES = {
    'C': (np.array([12.0, 13.00335]),
          np.array([0.9893, 0.0107])),
    'H': (np.array([1.00783, 2.01410]),
          np.array([0.999885, 0.000115])),
    'N': (np.array([14.00307, 15.00011]),
          np.array([0.99636, 0.00364])),
    'O': (np.array([15.99491, 16.99913, 17.99916]),
          np.array([0.99757, 0.00038, 0.00205])),
    'S': (np.array([31.97207, 32.97146, 33.96787, 35.96708]),
          np.array([0.9499, 0.0075, 0.0425, 0.0001])),
}

_AVERAGINE_STOICHIOMETRY = {
    'C': 4.9384, 'H': 7.7583, 'N': 1.3577, 'O': 1.4773, 'S': 0.0417,
}
_AVERAGINE_MASS = sum(
    n * _ISOTOPES[a][0][0] for a, n in _AVERAGINE_STOICHIOMETRY.items()
)  # ~111.1254 Da


def averagine_composition(mass):
    """Atomic composition (rounded to integers) of an averagine peptide of the
    given monoisotopic mass."""
    n = mass / _AVERAGINE_MASS
    return {a: max(int(round(n * x)), 0) for a, x in _AVERAGINE_STOICHIOMETRY.items()}


def natural_envelope(composition, n_peaks=30):
    """Natural isotope envelope (monoisotopic first), computed by repeated
    convolution of single-atom isotope distributions. Returns an array of
    length ≤ n_peaks (truncated). Indexed by integer mass offset in Da.
    """
    dist = np.array([1.0])
    for atom, count in composition.items():
        if count <= 0 or atom not in _ISOTOPES:
            continue
        _, abund = _ISOTOPES[atom]
        # Raise the atomic distribution to the count-th convolution power
        # via exponentiation by squaring.
        accum = np.array([1.0])
        base = abund.copy()
        k = int(count)
        while k > 0:
            if k & 1:
                accum = np.convolve(accum, base)[:n_peaks]
            k >>= 1
            if k:
                base = np.convolve(base, base)[:n_peaks]
        dist = np.convolve(dist, accum)[:n_peaks]
    dist = dist / dist.sum()
    return dist


def mass_envelope(uptake_dist, natural_env, n_peaks=None):
    """Convolve the uptake distribution with a natural isotope envelope.

    Parameters
    ----------
    uptake_dist : ndarray, shape (nT, nU+1)
    natural_env : ndarray, length M
    n_peaks : int or None
        Output number of peaks. Defaults to nU + M - 1.

    Returns
    -------
    env : ndarray, shape (nT, n_peaks)
    """
    uptake_dist = np.asarray(uptake_dist, dtype=float)
    natural_env = np.asarray(natural_env, dtype=float)
    nT, nU1 = uptake_dist.shape
    nU = nU1 - 1
    M = len(natural_env)
    L = nU + M - 1 if n_peaks is None else n_peaks
    out = np.zeros((nT, L))
    for ti in range(nT):
        conv = np.convolve(uptake_dist[ti], natural_env)
        out[ti, :min(L, len(conv))] = conv[:L]
    return out


# =====================================================================
# Plotting defaults
# =====================================================================

def set_plot_style(use_tex=False):
    """Set matplotlib rcParams to a publication-oriented style."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['cmr10', 'cmss10', 'cmtt10'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'axes.formatter.use_mathtext' : True,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.3,
        'lines.markersize': 3.5,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'mathtext.fontset': 'cm',
        'text.usetex': use_tex,
    })
    mpl.rcParams['axes.formatter.use_mathtext'] = True



# Okabe–Ito colour-blind safe palette (used consistently across regimes)
COLORS = {
    'uncoupled':   '#0072B2',   # blue
    'concerted':   '#D55E00',   # vermillion
    'basins':      '#009E73',   # green
    'full':        '#CC79A7',   # reddish purple
    'other1':      '#E69F00',   # orange
    'other2':      '#56B4E9',   # sky blue
    'other3':      '#F0E442',   # yellow
    'other4':      '#000000',
}

LABELS = {
    'uncoupled':    'uncoupled residues',
    'concerted':    'concerted switching',
    'basins':       'metastable basins',
    'full':         'full Markov chain',
}


__all__ = [
    'UncoupledResidues', 'ConcertedSwitching', 'MetastableBasins', 'FullMarkov',
    'uptake_from_survivals', 'moments_from_distribution',
    'random_singleflip_W', 'random_dense_W',
    'averagine_composition', 'natural_envelope', 'mass_envelope',
    'set_plot_style', 'COLORS', 'LABELS',
]
