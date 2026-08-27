"""RQ2 / H1: is a factored state better than one mutually exclusive label?

The proposal's RQ2 asks whether *overlapping market dimensions* can be estimated more
reliably than a single mutually exclusive regime label, and H1 claims a multidimensional
state is more temporally stable and more useful.

The comparison is only meaningful at **matched cardinality**. A 3x3 factored state and a
flat 9-state label describe exactly the same 9 joint cells, so neither is more expressive;
what differs is how many parameters each spends to get there:

    factored  2 x (3x3 transitions + 3 means + 3 vars)  =  30
    flat          9x9 transitions + 9x2 means + 9x2 vars = 117

Factoring buys parameter efficiency by *assuming the axes evolve independently*. That
assumption is false in markets — trend and volatility are related — so the test is whether
the efficiency is worth the misspecification out of sample. Compare at matched *parameters*
instead and the flat model is crippled by construction; compare without matching anything
and the result is a tautology about model size.
"""
from msl.factored.hmm_k import GaussHMM
from msl.factored.compare import factored_vs_flat, stability

__all__ = ["GaussHMM", "factored_vs_flat", "stability"]
