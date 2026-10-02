Discrete Gibbs Iterative Neural Sampler (arXiv 2610.01739), a small JAX slice: Gibbs labels, the Proposition 3.1 loss, any-order importance weights, and ESS tempering on a 2x2 open Ising lattice.

```
/home/nixos/arxiv-impl-work/bin/py -m pip install -e .
/home/nixos/arxiv-impl-work/bin/py -m pytest -q
```

Does not match the paper: no alloys, no section 3.3 transport, a tiny MLP, a 2x2 open lattice, CPU only.
