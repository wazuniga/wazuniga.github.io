# p-Adic Buice–Cowan theory: code, data and LaTeX source

Supplementary material for

* W. A. Zúñiga-Galindo, *p-Adic statistical field theory of fluctuation effects in
  hierarchical neural networks* (long paper).
* W. A. Zúñiga-Galindo, *Hierarchical connectivity sets the upper critical dimension of
  neural avalanches* (short paper).

The folder contains the simulation code of Secs. VIII–IX of the long paper (avalanches and
the small-world property), the scripts that draw the figures, the simulation data used in
the papers, and the LaTeX source of the long paper.

## Contents

```
papers/            PDF versions of the two papers
code/sim/          avalanche simulations (C) and their analysis/plotting (Python)
code/smallworld/   small-world tests: sparse graphs and spreading times (C, Python)
code/figures/      scripts for the analytic figures (connection matrices, spectrum,
                   propagator, regional networks)
data/sim/          avalanche data (.npz, .npy), fitted exponents (summary*.json),
                   logs of the critical-point searches (*.log)
data/smallworld/   spreading times (cov_*.txt, lv_*.txt) and graph statistics (*.json)
latex/             LaTeX source of the long paper (arXiv package: .tex, REVTeX files, figures)
```

## Requirements

* A C compiler (gcc or clang) and the C math library.
* Python 3 with NumPy, SciPy and Matplotlib.

## Regenerating the figures from the stored data

Each plotting script reads its data from, and writes its figure to, the current directory.

```
cd data/sim
python3 ../../code/sim/plot_aval.py        # fig_avalanches.pdf  (avalanche-size distributions)
python3 ../../code/sim/lp_test.py          # lp_result.json      (log-periodicity test)
python3 ../../code/sim/plot_lp.py          # fig_logperiodic.pdf

cd ../smallworld
python3 ../../code/smallworld/plot_sw.py   # fig_smallworld.pdf

mkdir -p ../../figs && cd ../../figs
python3 ../code/figures/make_figs.py       # connection matrices, level spectrum, propagator
python3 ../code/figures/make_fig_regional.py
```

## Rerunning the simulations

The simulations are exact in continuous time (event-driven, Gillespie-type algorithm
with thinning). The C programs write raw results to the system temporary directory; the
Python drivers call them as `./aval`, `./avalp`, etc., so build them in the directory where
you run the drivers:

```
cd code/sim
gcc -O2 -o aval  aval.c  -lm      # p = 2 tree, chain, branching process, power-law chain
gcc -O2 -o avalp avalp.c -lm      # trees with general p
python3 production.py             # critical points and avalanche statistics (Table II)
bash levy_camp.sh                 # Euclidean chain with power-law coupling
python3 analyze.py                # exponents -> summary.json
bash camp4.sh                     # per-seed runs for the log-periodicity test (p = 7)

cd ../smallworld
gcc -O2 -o cover cover.c -lm      # spreading time on the tree and on the ring
gcc -O2 -o levy  levy.c  -lm      # spreading time on the ring with power-law coupling
bash camp.sh; bash camp2.sh       # spreading-time campaigns
python3 graph.py; python3 graph2.py   # sparse-graph statistics
```

The full campaigns take many CPU hours. The command-line arguments of the C programs are
documented at the top of each `main` (for example
`./aval mode sigma b s0 nruns tmax seed L outfile [nmax]`).

## Model and units

Effective spike model of Buice and Cowan with firing-rate function
f(s) = f_max (1 − exp(−s/s0)), s0 = 0.25. Units: decay rate α = 1 and total coupling
w0 = 1, so the control parameter is b = f'(0) w0 / α. On the tree (p = 2, depth 40) the
coupling between two neurons whose smallest common ball has 2^k neurons is proportional
to 2^(−k(1+σ)).

## Note on the code

As stated in the Methods of both papers, the author used Claude (Anthropic) to help
implement the numerical simulation code.

## Contact

W. A. Zúñiga-Galindo, University of Texas Rio Grande Valley.
