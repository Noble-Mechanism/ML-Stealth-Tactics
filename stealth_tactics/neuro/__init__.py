"""Spec 6: neural policy and neuroevolution (docs/specs/06-neural-policy-neuroevolution.md).

- ``mlp``: ``Arch``, ``MLPPolicy`` (flat 231-64-64-13 or the per-contact set
  encoder behind ``encoder="set"``), init, flat <-> layers.
- ``genome``: ``NetGenome`` (weights per network, own sigma, lineage, network
  count), self-adaptive Gaussian mutation, unit swap (off), save / load.
- ``novelty``: behaviour descriptor, novelty, archive, hall of fame.
- ``neuroga``: ``NeuroConfig``, ``NeuroGA`` (loop, selection, elites,
  benchmark, stagnation, checkpoints, final test).
- ``clone``: behaviour cloning of ``HandBlue`` (numpy Adam) + quality gate.
- ``checkpoint``: deterministic npz + atomic writes.
"""
