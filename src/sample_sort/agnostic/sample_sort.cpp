// Topology-agnostic sample sort over MPI.
// Owner: Sonok Mahapatra (CMSC 714 group project).
//
// Pipeline (see ../DESIGN.md for full rationale):
//   1. local sort
//   2. regular sample (p-1 samples per rank)
//   3. allgather samples
//   4. pick p-1 global splitters
//   5. partition local data by splitters -> send_counts[p]
//   6. alltoall send_counts -> recv_counts
//   7. alltoallv to redistribute the data itself
//   8. local sort/merge of received bucket
//
// Usage: see print_usage() below or run with --help.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Key = int64_t;
static const MPI_Datatype MPI_KEY = MPI_INT64_T;

static void print_usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " [options] [TOTAL_N]\n"
      "\n"
      "Sort TOTAL_N int64 keys distributed evenly across MPI ranks.\n"
      "Default TOTAL_N = 16777216 (1<<24). Must be divisible by ranks.\n"
      "\n"
      "Options:\n"
      "  --per-rank-N=N   Use N keys per rank (sets TOTAL_N = N * num_ranks).\n"
      "                   For weak-scaling runs.\n"
      "  --seed=S         RNG seed (default 0 = rank-derived).\n"
      "  --verify         Verify result is globally sorted (slow; for dev).\n"
      "  -h, --help       Print this help and exit.\n";
}

// If `s` starts with `prefix`, fills `value` with the suffix and returns true.
static bool starts_with(const std::string& s, const char* prefix, std::string& value) {
    size_t plen = std::strlen(prefix);
    if (s.size() < plen || s.compare(0, plen, prefix) != 0) return false;
    value = s.substr(plen);
    return true;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // ---- Argument parsing ----
    long long total_N    = (1LL << 24);
    long long per_rank_N = -1;     // sentinel: not set
    uint64_t  seed       = 0;      // 0 = derive from rank
    bool      verify     = false;
    bool      saw_pos_N  = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        std::string val;
        if (a == "-h" || a == "--help") {
            if (rank == 0) print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (starts_with(a, "--per-rank-N=", val)) {
            per_rank_N = std::atoll(val.c_str());
        } else if (starts_with(a, "--seed=", val)) {
            seed = std::strtoull(val.c_str(), nullptr, 10);
        } else if (a == "--verify") {
            verify = true;
        } else if (!a.empty() && a[0] != '-') {
            total_N = std::atoll(a.c_str());
            saw_pos_N = true;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << a << "\n";
                print_usage(argv[0]);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    if (per_rank_N > 0 && saw_pos_N) {
        if (rank == 0)
            std::cerr << "Error: pass either TOTAL_N or --per-rank-N=, not both.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (per_rank_N > 0) total_N = per_rank_N * p;

    if (total_N <= 0 || total_N % p != 0) {
        if (rank == 0)
            std::cerr << "Error: total_N (" << total_N
                      << ") must be positive and divisible by ranks (" << p << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long local_n = total_N / p;

    // ---- Generate input data (reproducible per (seed, rank)) ----
    uint64_t rng_seed = seed
        ? (seed ^ (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL))
        : (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL + 1);
    std::vector<Key> local(local_n);
    std::mt19937_64 rng(rng_seed);
    for (auto& x : local) x = static_cast<Key>(rng());

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // ===== STEP 1: local sort =====
    // TODO: std::sort(local.begin(), local.end());

    // ===== STEP 2: regular sample (pick p-1 samples) =====
    std::vector<Key> samples;  // size should end up as p-1
    // TODO: indices to sample at: i * local_n / p for i = 1 .. p-1.

    // ===== STEP 3: allgather samples =====
    std::vector<Key> all_samples(static_cast<size_t>(p) * (p - 1));
    // TODO: MPI_Allgather(samples.data(), p - 1, MPI_KEY,
    //                     all_samples.data(), p - 1, MPI_KEY,
    //                     MPI_COMM_WORLD);

    // ===== STEP 4: pick p-1 global splitters =====
    std::vector<Key> splitters;  // size p-1
    // TODO: std::sort(all_samples.begin(), all_samples.end());
    //       splitters[i] = all_samples[(i+1) * p - 1] for i = 0..p-2
    //       (one valid evenly-spaced rule — see DESIGN.md).

    // ===== STEP 5: partition local data by splitters =====
    std::vector<int> send_counts(p, 0);
    std::vector<int> send_displs(p, 0);
    // TODO: walk `local` (or use std::lower_bound on `splitters`) and fill
    //       send_counts[i] = # of local keys whose target rank is i.
    //       send_displs[i] = exclusive prefix sum of send_counts.

    // ===== STEP 6: exchange counts =====
    std::vector<int> recv_counts(p, 0);
    // TODO: MPI_Alltoall(send_counts.data(), 1, MPI_INT,
    //                    recv_counts.data(), 1, MPI_INT,
    //                    MPI_COMM_WORLD);

    // ===== STEP 7: redistribute via Alltoallv =====
    std::vector<int> recv_displs(p, 0);
    long long recv_total = 0;
    std::vector<Key> received;
    // TODO: build recv_displs (exclusive prefix sum of recv_counts);
    //       recv_total = sum of recv_counts; received.resize(recv_total);
    //       MPI_Alltoallv(local.data(),    send_counts.data(), send_displs.data(), MPI_KEY,
    //                     received.data(), recv_counts.data(), recv_displs.data(), MPI_KEY,
    //                     MPI_COMM_WORLD);

    // ===== STEP 8: local sort/merge of received bucket =====
    // TODO: std::sort(received.begin(), received.end());
    //       (Each contiguous chunk is already sorted; in-place k-way merge
    //        would be faster. Defer.)

    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Sample sort time: " << (t_end - t_start) << " s\n";

    // ---- Optional verification (does not count toward timing) ----
    if (verify) {
        // TODO: implement.
        //   1. Check `received` is locally sorted.
        //   2. Send received.front() to rank-1 and received.back() to rank+1
        //      via MPI_Sendrecv; check rank r's max <= rank r+1's min.
        //   3. Check sum of recv_counts across all ranks equals total_N.
        // Print "VERIFY: ok" or "VERIFY: FAIL: <reason>" on rank 0.
        if (rank == 0) std::cerr << "VERIFY: not yet implemented\n";
    }

    MPI_Finalize();
    return 0;
}
