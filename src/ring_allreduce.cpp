#include <mpi.h>

#include <iostream>
#include <vector>

constexpr int VECTOR_SIZE = 1 << 20;  // 1M doubles ~ 8MB, typical ML gradient

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> data(VECTOR_SIZE, static_cast<double>(rank));
    std::vector<double> buffer(VECTOR_SIZE, 0.0);

    const int chunk = VECTOR_SIZE / size;
    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;

    double t_start = MPI_Wtime();

    // Phase 1: Scatter-reduce — each step accumulates one chunk
    for (int step = 0; step < size - 1; step++) {
        int send_chunk = (rank - step + size) % size;
        int recv_chunk = (rank - step - 1 + size) % size;

        MPI_Sendrecv(data.data() + send_chunk * chunk, chunk, MPI_DOUBLE, next, 0,
                     buffer.data() + recv_chunk * chunk, chunk, MPI_DOUBLE, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = recv_chunk * chunk; i < (recv_chunk + 1) * chunk; i++)
            data[i] += buffer[i];
    }

    // Phase 2: Allgather — propagate fully reduced chunks around the ring
    for (int step = 0; step < size - 1; step++) {
        int send_chunk = (rank - step + 1 + size) % size;
        int recv_chunk = (rank - step + size) % size;

        MPI_Sendrecv(data.data() + send_chunk * chunk, chunk, MPI_DOUBLE, next, 0,
                     data.data() + recv_chunk * chunk, chunk, MPI_DOUBLE, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "RingAllreduce time: "
                  << (t_end - t_start)
                  << " s\n";

    MPI_Finalize();
    return 0;
}