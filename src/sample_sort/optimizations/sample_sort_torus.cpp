// Torus-optimized sample sort (Phase 3 — topology-aware variant).
// Owner: Sonok Mahapatra.
//
// Baseline to beat (from ../RESULTS.md):
//   strong np=256: 0.0069 s
//   weak   np=256: 0.0488 s
//
// Optimization idea: treat ranks as a 2D grid of side sqrt(p). The global
// p-way alltoallv decomposes into two phases of sqrt(p)-way alltoallvs —
// row-wise then column-wise — each on smaller sub-communicators. Each
// sub-alltoallv moves only sqrt(p)-way data over fewer torus hops.
//
// Constraint: p must be a perfect square (16, 64, 256 satisfy this; 32, 128
// do not, so the matching-topology runs at np=32, 128 will need a
// fallback or a non-square 2D grid).
//
// This file is a SCAFFOLD: it currently runs the agnostic algorithm. Step 7
// is marked as the replacement point.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Key = int64_t;
static const MPI_Datatype MPI_KEY = MPI_INT64_T;

static void print_usage(const char* prog) {
    std::cerr <<
      "Usage: " << prog << " [options] [TOTAL_N]\n"
      "Torus-optimized sample sort. Same CLI as the agnostic variant.\n"
      "  --per-rank-N=N   N keys per rank (TOTAL_N = N * num_ranks).\n"
      "  --seed=S         RNG seed (default 0 = rank-derived).\n"
      "  --verify         Verify globally sorted (slow; for dev).\n"
      "  -h, --help       Print this help and exit.\n";
}

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

    long long total_N    = (1LL << 24);
    long long per_rank_N = -1;
    uint64_t  seed       = 0;
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
        if (rank == 0) std::cerr << "Error: pass either TOTAL_N or --per-rank-N=, not both.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (per_rank_N > 0) total_N = per_rank_N * p;
    if (total_N <= 0 || total_N % p != 0) {
        if (rank == 0)
            std::cerr << "Error: total_N (" << total_N << ") must be positive and divisible by ranks ("
                      << p << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    long long local_n = total_N / p;
    if (local_n < p) {
        if (rank == 0) std::cerr << "Error: local_n (" << local_n << ") must be >= ranks (" << p << ").\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    uint64_t rng_seed = seed
        ? (seed ^ (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL))
        : (static_cast<uint64_t>(rank) * 0x9E3779B97F4A7C15ULL + 1);
    std::vector<Key> local(local_n);
    std::mt19937_64 rng(rng_seed);
    for (auto& x : local) x = static_cast<Key>(rng());

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // ===== Steps 1-6: identical to agnostic =====
    std::sort(local.begin(), local.end());

    std::vector<Key> samples(p - 1);
    for (int i = 0; i < p - 1; i++)
        samples[i] = local[(i + 1) * local_n / p];

    std::vector<Key> all_samples(static_cast<size_t>(p) * (p - 1));
    MPI_Allgather(samples.data(),     p - 1, MPI_KEY,
                  all_samples.data(), p - 1, MPI_KEY,
                  MPI_COMM_WORLD);

    std::sort(all_samples.begin(), all_samples.end());
    std::vector<Key> splitters(p - 1);
    for (int i = 0; i < p - 1; i++)
        splitters[i] = all_samples[(i + 1) * p - 1];

    std::vector<int> send_counts(p, 0);
    std::vector<int> send_displs(p, 0);
    {
        long long prev = 0;
        for (int i = 0; i < p - 1; i++) {
            auto it = std::lower_bound(local.begin() + prev, local.end(), splitters[i]);
            long long next = it - local.begin();
            send_counts[i] = static_cast<int>(next - prev);
            send_displs[i] = static_cast<int>(prev);
            prev = next;
        }
        send_counts[p - 1] = static_cast<int>(local_n - prev);
        send_displs[p - 1] = static_cast<int>(prev);
    }

    std::vector<int> recv_counts(p, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // ===== STEP 7 (TORUS OPTIMIZATION POINT) =====
    //
    // Replace the global MPI_Alltoallv with row-then-column phases on a 2D
    // process grid:
    //
    //   1. Build a 2D Cartesian comm: MPI_Cart_create with dims = {sqrt(p), sqrt(p)}
    //   2. Extract row + column sub-comms via MPI_Cart_sub.
    //   3. Reorganize send_counts so chunks for the same row-target are
    //      contiguous, then do MPI_Alltoallv on the row sub-comm.
    //   4. Reorganize the intermediate buffer by column-target.
    //   5. Do a second MPI_Alltoallv on the column sub-comm.
    //
    // Why this helps on a torus: each phase moves only sqrt(p)-way data, and
    // each row/column maps onto a 1D torus row/column where messages travel
    // at most sqrt(p)/2 hops instead of p/2.
    //
    // Constraint check: p must be a perfect square. For np in {16,64,256}
    // this works. For np in {32,128} you'll need either:
    //   - a non-square 2D grid (e.g. 4x8 for np=32)
    //   - or fall back to the agnostic alltoallv for those np values
    //
    // TODO: replace the placeholder MPI_Alltoallv below with the row+column
    //       phases.

    std::vector<Key> received;

    // ----- 2-phase Torus-aware alltoallv: row sub-alltoallv then column sub-alltoallv -----
    // Build a 2D Cartesian comm with balanced dims via MPI_Dims_create.
    int dims[2] = {0, 0};
    MPI_Dims_create(p, 2, dims);   // dims[0] * dims[1] == p
    int periods[2] = {1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /*reorder=*/0, &cart_comm);

    // (Cart coords exist; we don't need myrow/mycol explicitly because the
    //  redistribution logic indexes into send_counts via the same row-major
    //  rank == r*dims[1] + c convention used by MPI_Cart_create.)

    // Sub-comms. With cart_create row-major: rank == row*dims[1] + col.
    int remain_row[2] = {0, 1};   // drop dim 0 → "row sub-comm" (varying col)
    int remain_col[2] = {1, 0};   // drop dim 1 → "col sub-comm" (varying row)
    MPI_Comm row_comm, col_comm;
    MPI_Cart_sub(cart_comm, remain_row, &row_comm);
    MPI_Cart_sub(cart_comm, remain_col, &col_comm);

    // Build per-row-mate per-dest-row count matrix (we need to know, for each
    // row-mate c' we'll send to in phase 1, how that bundle further breaks
    // down by destination row). counts2d[c' * dims[0] + r] = send_counts[r*dims[1]+c']
    std::vector<int> my_counts2d(static_cast<size_t>(dims[1]) * dims[0]);
    for (int c = 0; c < dims[1]; c++)
        for (int r = 0; r < dims[0]; r++)
            my_counts2d[c * dims[0] + r] = send_counts[r * dims[1] + c];

    // Phase 1 send_counts/displs (for row sub-comm): per-target-col aggregate
    std::vector<int> p1_send_counts(dims[1], 0);
    std::vector<int> p1_send_displs(dims[1], 0);
    int total_p1_send = 0;
    for (int c = 0; c < dims[1]; c++) {
        for (int r = 0; r < dims[0]; r++)
            p1_send_counts[c] += send_counts[r * dims[1] + c];
        p1_send_displs[c] = total_p1_send;
        total_p1_send += p1_send_counts[c];
    }

    // Pack p1_send_buf: for each col c', concatenate chunks for (r, c') for r in 0..dims[0]-1
    std::vector<Key> p1_send_buf(total_p1_send);
    {
        std::vector<int> wptr(p1_send_displs);
        for (int r = 0; r < dims[0]; r++) {
            for (int c = 0; c < dims[1]; c++) {
                int dest = r * dims[1] + c;
                int sz = send_counts[dest];
                std::copy(local.begin() + send_displs[dest],
                          local.begin() + send_displs[dest] + sz,
                          p1_send_buf.begin() + wptr[c]);
                wptr[c] += sz;
            }
        }
    }

    // Exchange the per-(c', r) count matrix on the row sub-comm so we can
    // unpack phase-1 receipts by destination row in phase 2.
    std::vector<int> recv_counts2d(static_cast<size_t>(dims[1]) * dims[0]);
    MPI_Alltoall(my_counts2d.data(), dims[0], MPI_INT,
                 recv_counts2d.data(), dims[0], MPI_INT,
                 row_comm);
    // recv_counts2d[c' * dims[0] + r] = bytes from row-mate c' destined for (r, mycol).

    // Phase 1 recv_counts (per row-mate sum)
    std::vector<int> p1_recv_counts(dims[1], 0);
    std::vector<int> p1_recv_displs(dims[1], 0);
    int total_p1_recv = 0;
    for (int c = 0; c < dims[1]; c++) {
        for (int r = 0; r < dims[0]; r++)
            p1_recv_counts[c] += recv_counts2d[c * dims[0] + r];
        p1_recv_displs[c] = total_p1_recv;
        total_p1_recv += p1_recv_counts[c];
    }

    // Phase 1 alltoallv on row sub-comm
    std::vector<Key> p1_recv_buf(total_p1_recv);
    MPI_Alltoallv(p1_send_buf.data(), p1_send_counts.data(), p1_send_displs.data(), MPI_KEY,
                  p1_recv_buf.data(), p1_recv_counts.data(), p1_recv_displs.data(), MPI_KEY,
                  row_comm);

    // Reorganize p1_recv_buf (currently grouped by source-col, inner by dest-row)
    // into p2_send_buf (grouped by dest-row, inner by source-col).
    std::vector<int> p2_send_counts(dims[0], 0);
    std::vector<int> p2_send_displs(dims[0], 0);
    int total_p2_send = 0;
    for (int r = 0; r < dims[0]; r++) {
        for (int c = 0; c < dims[1]; c++)
            p2_send_counts[r] += recv_counts2d[c * dims[0] + r];
        p2_send_displs[r] = total_p2_send;
        total_p2_send += p2_send_counts[r];
    }

    std::vector<Key> p2_send_buf(total_p2_send);
    {
        std::vector<int> wptr(p2_send_displs);
        int rptr = 0;
        for (int c = 0; c < dims[1]; c++) {
            for (int r = 0; r < dims[0]; r++) {
                int sz = recv_counts2d[c * dims[0] + r];
                std::copy(p1_recv_buf.begin() + rptr,
                          p1_recv_buf.begin() + rptr + sz,
                          p2_send_buf.begin() + wptr[r]);
                rptr += sz;
                wptr[r] += sz;
            }
        }
    }

    // Phase 2 recv counts via alltoall on col sub-comm
    std::vector<int> p2_recv_counts(dims[0], 0);
    MPI_Alltoall(p2_send_counts.data(), 1, MPI_INT,
                 p2_recv_counts.data(), 1, MPI_INT,
                 col_comm);
    std::vector<int> p2_recv_displs(dims[0], 0);
    int total_p2_recv = 0;
    for (int r = 0; r < dims[0]; r++) {
        p2_recv_displs[r] = total_p2_recv;
        total_p2_recv += p2_recv_counts[r];
    }

    received.resize(static_cast<size_t>(total_p2_recv));
    MPI_Alltoallv(p2_send_buf.data(), p2_send_counts.data(), p2_send_displs.data(), MPI_KEY,
                  received.data(),    p2_recv_counts.data(), p2_recv_displs.data(), MPI_KEY,
                  col_comm);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    // -----------------------------------------------------------------------------------

    // ===== Step 8: identical to agnostic =====
    std::sort(received.begin(), received.end());

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Sample sort time: " << (t_end - t_start) << " s\n";

    if (verify) {
        int local_ok = std::is_sorted(received.begin(), received.end()) ? 1 : 0;
        int all_local_ok = 0;
        MPI_Reduce(&local_ok, &all_local_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);

        Key my_min = received.empty() ? std::numeric_limits<Key>::max() : received.front();
        Key my_max = received.empty() ? std::numeric_limits<Key>::min() : received.back();
        int has_data = received.empty() ? 0 : 1;

        std::vector<Key> all_mins, all_maxs;
        std::vector<int> all_has;
        if (rank == 0) { all_mins.resize(p); all_maxs.resize(p); all_has.resize(p); }
        MPI_Gather(&my_min,   1, MPI_KEY, all_mins.data(), 1, MPI_KEY, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_max,   1, MPI_KEY, all_maxs.data(), 1, MPI_KEY, 0, MPI_COMM_WORLD);
        MPI_Gather(&has_data, 1, MPI_INT, all_has.data(),  1, MPI_INT, 0, MPI_COMM_WORLD);

        long long my_count = static_cast<long long>(received.size());
        long long total_count = 0;
        MPI_Reduce(&my_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            bool monotonic = true;
            Key last_max = std::numeric_limits<Key>::min();
            bool seen = false;
            for (int r = 0; r < p; r++) {
                if (!all_has[r]) continue;
                if (seen && last_max > all_mins[r]) { monotonic = false; break; }
                last_max = all_maxs[r];
                seen = true;
            }
            bool count_ok = (total_count == total_N);
            if (all_local_ok && monotonic && count_ok) {
                std::cerr << "VERIFY: ok\n";
            } else {
                std::cerr << "VERIFY: FAIL  local_sorted=" << all_local_ok
                          << "  monotonic=" << monotonic
                          << "  total_count=" << total_count
                          << "  expected=" << total_N << "\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
