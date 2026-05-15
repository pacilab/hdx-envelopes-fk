"""
test
verify all regimes give consistent results and that the uncoupled residue
regime is reproduced by a Full Markov with single-flip product rates.
"""
import numpy as np
import hdx_env as H

rng = np.random.default_rng(0)
N = 4
P = tuple(range(N))

k_op = rng.uniform(0.5, 2.0, N)
k_cl = rng.uniform(0.5, 2.0, N)
k_int = rng.uniform(0.5, 2.0, N)
t = np.logspace(-2, 2, 25)

# Independent reference
ind = H.UncoupledResidues(k_op, k_cl, k_int)

# Build a FullMarkov with the same per-site uncoupled dynamics by
# constructing W as the Kronecker sum of per-site 2x2 generators promoted
# to the full space.
dim = 2 ** N
W_full = np.zeros((dim, dim))
for s in range(dim):
    for i in range(N):
        sp = s ^ (1 << i)
        bit = (s >> i) & 1
        # If bit=0, the residue closes->opens with rate k_op; if bit=1, opens->closes with rate k_cl
        W_full[s, sp] = k_op[i] if bit == 0 else k_cl[i]
full = H.FullMarkov(W_full, k_int)

# Compare single-site survivals
print("single-site survival (residue 0) — independent vs full:")
psi_ind = ind.psi_i(0, t)
psi_full = full.psi_X((0,), t)
max_err = np.max(np.abs(psi_ind - psi_full))
print("  max |diff|:", max_err)
assert max_err < 1e-8

# Compare pair survivals
psi2_ind = ind.psi_X((0, 1), t)
psi2_full = full.psi_X((0, 1), t)
err2 = np.max(np.abs(psi2_ind - psi2_full))
print("pair survival {0,1} — max|diff|:", err2)
assert err2 < 1e-8

# Compare full uptake distributions
ud_ind = ind.uptake_distribution(P, t)
ud_full = full.uptake_distribution(P, t)
err_ud = np.max(np.abs(ud_ind - ud_full))
print("uptake distribution — max|diff|:", err_ud)
assert err_ud < 1e-8

# Concerted regime sanity: in the limit of very fast switching, it should
# behave like independent residues with effective rate pi_q * k_int_i
k_op_q, k_cl_q = 1e4, 1e4
conc = H.ConcertedSwitching(k_op_q, k_cl_q, k_int)
ud_c = conc.uptake_distribution(P, t)
pi_q = k_op_q / (k_op_q + k_cl_q)
ind_eff = H.UncoupledResidues(np.full(N, k_op_q),
                                np.full(N, k_cl_q),
                                k_int)  # keeps per-site open-prob = pi_q
# Fast-switch (EX2) limit: psi_i(t) ≈ exp(-pi_q k_int_i t)
psi_expected = np.exp(-pi_q * k_int[0] * t)
psi_conc_single = conc.psi_X((0,), t)
err_fast = np.max(np.abs(psi_conc_single - psi_expected))
print("concerted (fast-switch) single-site vs EX2 — max|diff|:", err_fast)
assert err_fast < 1e-3

# Basins frozen sanity: a single basin reproduces independent residues
# with rates pi*k_int
pi_one = rng.uniform(0.05, 0.95, (1, N))
basins = H.MetastableBasins(pi_one, k_int)
ud_b = basins.uptake_distribution(P, t)
# In each site-survival is exp(-k_int_i * pi_open_i t)
psi_site_expected = np.exp(-k_int[0] * pi_one[0, 0] * t)
psi_b = basins.psi_X((0,), t)
err_b = np.max(np.abs(psi_site_expected - psi_b))
print("basins (frozen, B=1) single-site — max|diff|:", err_b)
assert err_b < 1e-12

# Natural envelope sanity
comp = H.averagine_composition(1000.0)
env = H.natural_envelope(comp, n_peaks=15)
print("natural envelope (1000 Da) first 6 peaks:", env[:6].round(4))
assert abs(env.sum() - 1.0) < 1e-9

# Mass envelope convolution
ud = ind.uptake_distribution(P, t[:3])
me = H.mass_envelope(ud, env)
print("mass-envelope shape:", me.shape)
print("row sums:", me.sum(axis=1).round(6))

print("\nAll checks passed.")
