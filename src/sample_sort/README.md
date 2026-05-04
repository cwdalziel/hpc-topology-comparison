# Sample Sort (CMSC 714)

Sample-sort component of the topology-vs-MPI-workload comparison study.
**Owner:** Sonok Mahapatra.

## Layout

- `agnostic/sample_sort.cpp` — topology-agnostic baseline. Standard MPI
  collectives only, no topology assumptions.
- `optimizations/` — one variant per topology (ring, torus, hypercube,
  fat_tree, dragonfly), added once the baseline is working.
- `DESIGN.md` — algorithm pipeline, key-type rationale, per-topology
  optimization sketches, and open questions.

## Build

The top-level `Makefile` auto-discovers any `*.cpp` under `src/`. From the
repo root:

```
make
```

This produces `bin/sample_sort/agnostic/sample_sort` (and binaries for any
optimization variants once they exist, e.g.
`bin/sample_sort/optimizations/sample_sort_ring`).

## Run

### One config across all topologies

```
# from repo root
./run_benchmarks.sh bin/sample_sort/agnostic/sample_sort <np> <total_N>
```

`total_N` must be divisible by `<np>`. Each invocation sweeps every topology
XML in `platforms/<np>/` and writes one CSV row per topology to
`results/<np>/sample_sort_results.csv`.

### Full strong + weak sweep

```
# from repo root
./run_sample_sort_scaling.sh [strong|weak|all]   # default: all
```

Loops `np ∈ {16,32,64,128,256}` × every sample-sort binary × every topology
for that `np`. Strong = fixed `STRONG_N` (default `1<<24`) for every np.
Weak = `WEAK_PER_RANK * np` (default `WEAK_PER_RANK=1<<20`). Override via env:

```
STRONG_N=$((1<<22)) WEAK_PER_RANK=$((1<<19)) ./run_sample_sort_scaling.sh all
```

Missing optimization binaries are skipped, so this script works now (with
just the agnostic baseline) and picks up variants automatically as you add
them. Output goes to `results/strong/<np>/...` and `results/weak/<np>/...`.

### Direct binary CLI (for ad-hoc testing)

```
smpirun -np <np> -platform <xml> ./bin/sample_sort/agnostic/sample_sort [options] [TOTAL_N]
```

| Flag | Effect |
|---|---|
| `TOTAL_N` (positional) | Total keys to sort. Default `1<<24`. Must be divisible by `np`. |
| `--per-rank-N=N` | Use `N` keys per rank (sets `TOTAL_N = N * np`). Useful for weak scaling. |
| `--seed=S` | RNG seed (default `0` = rank-derived). For reproducibility / variance runs. |
| `--verify` | Check the result is globally sorted (slow; for development). |
| `-h`, `--help` | Print usage. |

## Status

Agnostic baseline — algorithm steps to fill in:

- [ ] 1. Local sort
- [ ] 2. Regular sample (`p-1` per rank)
- [ ] 3. Allgather samples
- [ ] 4. Pick global splitters
- [ ] 5. Partition local data by splitters
- [ ] 6. `MPI_Alltoall` exchange counts
- [ ] 7. `MPI_Alltoallv` redistribute data
- [ ] 8. Local merge / sort received bucket

Then:

- [ ] Benchmark agnostic baseline across all 5 topologies × `np ∈ {16,32,64,128,256}`
- [ ] Per-topology optimizations (5 variants)
- [ ] Re-benchmark optimized variants and compare
