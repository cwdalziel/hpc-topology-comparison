# Sample Sort — Design Notes

Working notes for the sample-sort sub-project. Treat this file as a living
decision log — append to it as things firm up.

## Algorithm pipeline (agnostic baseline)

Standard regular-sampling sample sort:

1. **Local sort.** Each rank sorts its `local_n = total_N / p` keys.
2. **Regular sample.** Each rank picks `p - 1` samples at evenly-spaced
   positions in its sorted local array.
3. **Allgather samples.** Every rank ends up with `p * (p - 1)` samples.
4. **Pick splitters.** Sort the gathered samples; pick `p - 1` global
   splitters at evenly-spaced positions. These define `p` buckets.
5. **Partition.** Each rank uses the splitters to split its local data into
   `p` buckets, producing `send_counts[p]`.
6. **Exchange counts.** `MPI_Alltoall` on `send_counts` so every rank knows
   how much it will receive.
7. **Redistribute.** `MPI_Alltoallv` exchanges the actual data. Bucket `i`
   ends up on rank `i`.
8. **Local merge / sort.** Each rank sorts its received bucket. (Merge is
   theoretically faster — incoming sub-arrays are each already sorted —
   but `std::sort` is fine for the first pass.)

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Key type | `int64_t` | Standard for parallel-sort benchmarks. 8-byte keys give meaningful all-to-all message sizes (the topology-sensitive step), without forcing floating-point semantics. |
| Input arg | `total_N` (global, not per-rank) | Matches the convention used by `3d_stencil` binaries. Strong vs. weak scaling becomes a question of how `total_N` varies with `np` in the driver, not a code change. |
| Divisibility | `total_N % p == 0` required | Clean per-rank partitioning. Same constraint Carson's ring stencil has. |
| Sampling | Regular (deterministic, evenly-spaced) | Simpler than random sampling and good enough for uniform inputs. Revisit if bucket imbalance shows up in results. |
| Timing scope | After data generation, with a barrier | Data setup is not the workload under test. |
| Output line | `Sample sort time: <T> s` on rank 0 | Matches the regex in `run_benchmarks.sh`. |

## Per-topology optimization plans (rough sketches)

To be expanded as we tackle each one. Don't hold any of these as committed
designs yet.

- **Ring** — global all-to-all is the worst case here; every message
  potentially traverses many hops. Replace with a pipeline of pairwise
  neighbor exchanges (odd-even or shift-by-k), accepting that each rank
  receives data from progressively further neighbors over many rounds.
- **Torus** — 2D process grid; do row-wise then column-wise sub-all-to-alls
  in row/column sub-communicators. Each phase moves less data over fewer
  hops.
- **Hypercube** — recursive halving / hyperquicksort: at each of `log p`
  steps, split the active group into two halves around a pivot and exchange
  across the matching hypercube dimension.
- **Fat tree** — keep the global `MPI_Alltoallv` (fat tree handles it
  well); the optimization win is from overlapping computation with
  communication, or chunking the all-to-all to pipeline it.
- **Dragonfly** — hierarchical: do an intra-group sample sort first, then
  a smaller inter-group redistribution. Minimizes traffic on the long
  global links.

## Open questions

- ~~**Strong vs. weak scaling.**~~ **Decided: do both.** Strong = fixed
  `total_N = STRONG_N` for every np (stresses the all-to-all — smaller
  buckets per rank, comm-bound). Weak = `total_N = WEAK_PER_RANK * np`
  (stresses pivot quality and per-rank sort cost). Both runs are wired up
  in `../../run_sample_sort_scaling.sh`; the binary itself accepts either
  positional `TOTAL_N` or `--per-rank-N=N` so you can hand-test either
  scaling regime without doing the multiplication yourself.
- **Skewed input distributions.** Random uniform keys are easy. Skewed
  distributions (e.g. Zipf) would surface load-imbalance differences
  across topologies, but probably out of scope for v1.
- **Verification.** Add a post-sort sanity check (per-rank `min/max` plus
  a global monotonicity check) during development; disable for benchmark
  runs so it doesn't pollute timings.
- **Local merge vs. local sort in step 8.** Empirically negligible until
  `np` is large; revisit if step 8 ends up being a non-trivial fraction
  of the runtime.
