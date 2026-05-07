// Dragonfly-optimized sample sort (Phase 3 — topology-aware variant).
// Owner: Sonok Mahapatra.
//
// Baseline to beat (from ../RESULTS.md):
//   strong np=256: 0.0092 s
//   weak   np=256: 0.1018 s
//
// Optimization idea: dragonfly groups ranks into "groups" connected by
// expensive global links. Hierarchically: do a sample sort first WITHIN
// each group (cheap, intra-group bandwidth is plentiful), then a smaller
// inter-group redistribution (uses the scarce global links once).
//
// This minimizes traffic on the long-haul links that are the dragonfly's
// primary cost driver.
//
// Practical: you'll need to know the group structure of the simulated
// dragonfly. Inspect platforms/<np>/dragonfly.xml — the group size depends
// on how the team authored the platform. Common choices: groups of size
// sqrt(p), or groups of fixed size with growing group count.
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
      "Dragonfly-optimized sample sort. Same CLI as the agnostic variant.\n"
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

    // ===== STEP 7 (DRAGONFLY OPTIMIZATION POINT) =====
    //
    // Replace the global MPI_Alltoallv with a hierarchical (group-aware)
    // redistribution:
    //
    //   Phase A (intra-group): within each group, do a local sample sort
    //                          / alltoallv on the group sub-comm.
    //   Phase B (inter-group): each group elects representatives (or all
    //                          ranks participate); redistribute keys whose
    //                          target rank is in another group, using the
    //                          smaller inter-group sub-comm.
    //
    // Rough plan:
    //   1. Determine group_size G and group_id for this rank.
    //      Read platforms/<np>/dragonfly.xml to confirm. If groups are
    //      structured as G nodes per group with p/G groups, then:
    //          group_id = rank / G
    //          intra_rank = rank % G
    //   2. Build sub-comms:
    //          MPI_Comm_split(MPI_COMM_WORLD, group_id, intra_rank,
    //                         &intra_group_comm);
    //          MPI_Comm_split(MPI_COMM_WORLD, intra_rank, group_id,
    //                         &inter_group_comm);
    //   3. Phase A: intra-group alltoallv (smaller scale, plentiful BW).
    //   4. Phase B: inter-group alltoallv (smaller scale, expensive links
    //                                       used only once).
    //
    // The exact algorithm varies — see PSRS-on-dragonfly or "k-level sample
    // sort" literature. The simplest version: sort intra-group first so each
    // group holds a roughly contiguous range, then do one inter-group
    // redistribute to align ranges with rank order.
    //
    // TODO: replace the placeholder MPI_Alltoallv below with the
    //       hierarchical redistribution. Inspect the platform XML before
    //       coding to lock down what the group size should be.

    std::vector<Key> received;

    // ----- 2-phase hierarchical alltoallv: intra-group then inter-group -----
    // Pick a group factorization. We do NOT parse platforms/<np>/dragonfly.xml
    // — instead we reuse MPI_Dims_create's balanced 2D factor, which gives us
    // dims = (num_groups, group_size). For the matching np values this is:
    //   p=16  -> 4x4    p=32  -> 4x8    p=64  -> 8x8
    //   p=128 -> 8x16   p=256 -> 16x16
    // The "groups" in dragonfly map to the first dim; intra-group ranks to
    // the second. A more exact match to the platform XML's group structure
    // would be more topology-aware; for now this still avoids the global
    // alltoallv pattern.
    int dims[2] = {0, 0};
    MPI_Dims_create(p, 2, dims);   // dims[0]=num_groups, dims[1]=group_size
    int num_groups  = dims[0];
    int group_size  = dims[1];
    int group_id    = rank / group_size;
    int intra_rank  = rank % group_size;

    MPI_Comm intra_group_comm, inter_group_comm;
    MPI_Comm_split(MPI_COMM_WORLD, group_id,   intra_rank, &intra_group_comm);  // size=group_size
    MPI_Comm_split(MPI_COMM_WORLD, intra_rank, group_id,   &inter_group_comm);  // size=num_groups

    // target rank == g*group_size + ir, where g = target_group, ir = intra-rank.
    // Phase 1 (intra-group): redistribute so chunks land on rank (g, intra_rank=ir)
    //   for ranks in MY group. After phase 1, intra_rank=ir holds chunks
    //   destined for (any_group, ir).
    // Phase 2 (inter-group): each rank now sends chunks destined for the right group.

    // counts2d[ir * num_groups + g] = bytes I send to rank (g, ir)
    std::vector<int> my_counts2d(static_cast<size_t>(group_size) * num_groups);
    for (int g = 0; g < num_groups; g++)
        for (int ir = 0; ir < group_size; ir++)
            my_counts2d[ir * num_groups + g] = send_counts[g * group_size + ir];

    // Phase 1 sends to intra-group rank ir the bundle for (any_group, ir).
    std::vector<int> p1_send_counts(group_size, 0);
    std::vector<int> p1_send_displs(group_size, 0);
    int total_p1_send = 0;
    for (int ir = 0; ir < group_size; ir++) {
        for (int g = 0; g < num_groups; g++)
            p1_send_counts[ir] += send_counts[g * group_size + ir];
        p1_send_displs[ir] = total_p1_send;
        total_p1_send += p1_send_counts[ir];
    }

    // Pack p1_send_buf grouped by intra_rank (ir), inner by group (g).
    std::vector<Key> p1_send_buf(total_p1_send);
    {
        std::vector<int> wptr(p1_send_displs);
        for (int g = 0; g < num_groups; g++) {
            for (int ir = 0; ir < group_size; ir++) {
                int dest = g * group_size + ir;
                int sz = send_counts[dest];
                std::copy(local.begin() + send_displs[dest],
                          local.begin() + send_displs[dest] + sz,
                          p1_send_buf.begin() + wptr[ir]);
                wptr[ir] += sz;
            }
        }
    }

    // Exchange the per-(ir, g) count matrix on intra_group_comm.
    std::vector<int> recv_counts2d(static_cast<size_t>(group_size) * num_groups);
    MPI_Alltoall(my_counts2d.data(),   num_groups, MPI_INT,
                 recv_counts2d.data(), num_groups, MPI_INT,
                 intra_group_comm);
    // recv_counts2d[ir * num_groups + g] = bytes from intra-group rank ir destined for (g, my_intra_rank).

    std::vector<int> p1_recv_counts(group_size, 0);
    std::vector<int> p1_recv_displs(group_size, 0);
    int total_p1_recv = 0;
    for (int ir = 0; ir < group_size; ir++) {
        for (int g = 0; g < num_groups; g++)
            p1_recv_counts[ir] += recv_counts2d[ir * num_groups + g];
        p1_recv_displs[ir] = total_p1_recv;
        total_p1_recv += p1_recv_counts[ir];
    }

    std::vector<Key> p1_recv_buf(total_p1_recv);
    MPI_Alltoallv(p1_send_buf.data(), p1_send_counts.data(), p1_send_displs.data(), MPI_KEY,
                  p1_recv_buf.data(), p1_recv_counts.data(), p1_recv_displs.data(), MPI_KEY,
                  intra_group_comm);
    // Free p1_send_buf and `local` — both no longer referenced. SMPI threads
    // share the process's memory; aggressive freeing matters at high np.
    std::vector<Key>().swap(p1_send_buf);
    std::vector<Key>().swap(local);

    // Reorganize phase-1 recv buf for phase 2: group by destination group g.
    std::vector<int> p2_send_counts(num_groups, 0);
    std::vector<int> p2_send_displs(num_groups, 0);
    int total_p2_send = 0;
    for (int g = 0; g < num_groups; g++) {
        for (int ir = 0; ir < group_size; ir++)
            p2_send_counts[g] += recv_counts2d[ir * num_groups + g];
        p2_send_displs[g] = total_p2_send;
        total_p2_send += p2_send_counts[g];
    }

    std::vector<Key> p2_send_buf(total_p2_send);
    {
        std::vector<int> wptr(p2_send_displs);
        int rptr = 0;
        for (int ir = 0; ir < group_size; ir++) {
            for (int g = 0; g < num_groups; g++) {
                int sz = recv_counts2d[ir * num_groups + g];
                std::copy(p1_recv_buf.begin() + rptr,
                          p1_recv_buf.begin() + rptr + sz,
                          p2_send_buf.begin() + wptr[g]);
                rptr += sz;
                wptr[g] += sz;
            }
        }
    }
    std::vector<Key>().swap(p1_recv_buf);

    std::vector<int> p2_recv_counts(num_groups, 0);
    MPI_Alltoall(p2_send_counts.data(), 1, MPI_INT,
                 p2_recv_counts.data(), 1, MPI_INT,
                 inter_group_comm);
    std::vector<int> p2_recv_displs(num_groups, 0);
    int total_p2_recv = 0;
    for (int g = 0; g < num_groups; g++) {
        p2_recv_displs[g] = total_p2_recv;
        total_p2_recv += p2_recv_counts[g];
    }

    received.resize(static_cast<size_t>(total_p2_recv));
    MPI_Alltoallv(p2_send_buf.data(), p2_send_counts.data(), p2_send_displs.data(), MPI_KEY,
                  received.data(),    p2_recv_counts.data(), p2_recv_displs.data(), MPI_KEY,
                  inter_group_comm);
    std::vector<Key>().swap(p2_send_buf);

    MPI_Comm_free(&intra_group_comm);
    MPI_Comm_free(&inter_group_comm);
    // -----------------------------------------------------------------------

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
