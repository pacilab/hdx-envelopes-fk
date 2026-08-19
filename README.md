# hdx-envelopes-fk

Simulation of hydrogen–deuterium exchange (HDX–MS) isotopic envelopes from reduced
models of conformational dynamics.

Companion code for:

> A. Grimaldi and E. Paci, *How conformational dynamics shape hydrogen–deuterium
> exchange isotopic envelopes* (2026).

## What it does

Four different pictures of conformational dynamics are implemented. They all map onto
the same peptide-level observable — the uptake distribution and the isotopic envelope —
through one common object, the family of **subset survival probabilities**

$$\psi_X(t) = \Pr[\text{no site in } X \text{ has exchanged by time } t], \qquad X \subseteq P.$$

Because every regime produces this same family of survivals, the envelope machinery —
uptake distribution, back exchange, and convolution with the natural isotope pattern —
is written once and shared by all of them.

Every figure in the paper is reproducible from notebooks `notebooks/fig1.ipynb` through `fig4.ipynb`
correspond to Figs. 1–4, and running one top to bottom regenerates that figure into a
`figures/` directory created on first run.

## Installation

```bash
conda env create -f environment.yml
conda activate hdxenv
```

## Quick start

```python
import numpy as np
import hdx_env as H
import hdx_backexchange as BX

N     = 8                      # amides in the peptide
P     = tuple(range(N))        # site indices 0..N-1
k_int = np.ones(N)             # intrinsic exchange rates
t     = np.array([0.0, 10.0, 100.0])

# Two regimes with identical single-residue rates
uncoupled = H.UncoupledResidues(np.full(N, 0.01), np.full(N, 0.1), k_int)
concerted = H.ConcertedSwitching(0.01, 0.1, k_int)

# Uptake distribution, shape (nT, N+1); rows sum to 1
ud = concerted.uptake_distribution(P, t)

# Same, but with back exchange: beta_i is the retention probability of site i
beta = np.full(N, 0.92)
prob_bx, prob_ideal, psi = BX.envelope_with_backexchange(concerted, P, t, beta)

# Convolve with the natural isotope pattern to get the observed envelope
nat = H.natural_envelope(H.averagine_composition(110.0 * N), n_peaks=15)
env = H.mass_envelope(prob_bx, nat, n_peaks=N + len(nat))

# The centroids agree between regimes; the widths do not
for name, m in (('uncoupled', uncoupled), ('concerted', concerted)):
    print(name, m.centroid(P, t).round(4), m.variance(P, t).round(4))
```

## Model classes

| class | picture | main parameters |
|---|---|---|
| `UncoupledResidues` | each amide an independent two-state Linderstrom-Lang site | `k_op`, `k_cl`, `k_int` (per site) |
| `ConcertedSwitching` | one shared binary switch gates all amides at once | `k_op_q`, `k_cl_q`, `k_int` |
| `MetastableBasins` | B basins with fast intrabasin equilibrium and slow (optionally frozen) interbasin transitions | `pi_open` (B×N), `p0_basin` or `W_basin` |
| `FullMarkov` | general continuous-time Markov chain on all $2^N$ states | `W` (generator), `k_int` |

Each one provides the same methods:

```
.psi_X(X, t)                  subset survival, shape (nT,)
.all_subset_survivals(P, t)   dict {tuple(X): array(nT)}
.uptake_distribution(P, t)    shape (nT, |P|+1)
.centroid(P, t) / .variance(P, t)
```

`UncoupledResidues` and `ConcertedSwitching` are the two limits that bracket the
behaviour; `MetastableBasins` and `FullMarkov` interpolate and generalise.

## Back exchange

`hdx_backexchange` applies back exchange *at the pattern level*: a site that is
deuterated at the end of labeling keeps its deuteron with probability $\beta_i$,
independently across sites. This is a substitution $z_i \to 1-\beta_i+\beta_i z_i$ in
the multivariate generating function.

## Intrinsic exchange rates

`python/kint` computes sequence-dependent intrinsic rates $k_{\rm int}$ .

> `python/kint` is the same code as the standalone repository
> <https://github.com/pacilab/hdx-rates-mixtures>, redistributed here under the same
> MIT license so that this repository is self-contained.
> . **If you use it, please cite the same work:**
>
> Grimaldi, A., Stofella, M., & Paci, E. (2026). Intrinsic Hydrogen–Deuterium Exchange
> Rates in H2O/D2O Mixtures. *The Journal of Physical Chemistry B*, **130**(9),
> 2493–2500. doi:[10.1021/acs.jpcb.5c06636](https://doi.org/10.1021/acs.jpcb.5c06636)
>
> That is a separate citation from the paper accompanying this repository; if you use
> both, please cite both. See [https://github.com/pacilab/hdx-rates-mixtures/blob/main/README.md](https://github.com/pacilab/hdx-rates-mixtures/blob/main/README.md).

```bash
cd python/kint
python kint.py --seq MYPEPTIDE --pH 7.0 --temp 298
```

| flag | meaning | default |
|---|---|---|
| `--seq` | sequence, literal string or path to a file | required |
| `--pH` | pH read on a glass electrode | required |
| `--temp` | absolute temperature (K) | required |
| `--deut` | solvent deuteration level, in [0, 1] | `1.0` |
| `--ref` | reference data, `PDLA` or `3Ala` | `PDLA` |
| `--time` | rate units, `h`, `m` or `s` | `s` |
| `--out` | output CSV path | auto-named |
| `--shift` | offset applied to the first residue index | `0` |

## Development note

This code was developed with the assistance of
[Claude Code](https://claude.com/claude-code) (Anthropic). The models and their
derivations are the authors'; numerical results were checked against the analytical
limits and cross-regime consistency tests collected in `python/_test.py`.

## License

MIT — see [LICENSE](LICENSE).
