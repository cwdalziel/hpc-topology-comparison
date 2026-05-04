# Sample Sort — Topology-Specific Optimizations

Topology-specific sample-sort variants live here, one `.cpp` per topology
(`sample_sort_ring.cpp`, `sample_sort_torus.cpp`, `sample_sort_hypercube.cpp`,
`sample_sort_fat_tree.cpp`, `sample_sort_dragonfly.cpp`).

These are added **after** the agnostic baseline in `../agnostic/` is working
and benchmarked, so we have a baseline to measure speedup against. See
`../DESIGN.md` for the planned strategy per topology.
