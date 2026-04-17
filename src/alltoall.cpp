#include <mpi.h>

#include <iostream>
#include <vector>

constexpr int MSG_SIZE = 1024;  // doubles per rank, try 256, 1024, 8192
constexpr int ITERATIONS = 10;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> sendbuf(size * MSG_SIZE, 1.0);
    std::vector<double> recvbuf(size * MSG_SIZE, 0.0);

    // Warmup
    MPI_Alltoall(sendbuf.data(), MSG_SIZE, MPI_DOUBLE,
                 recvbuf.data(), MSG_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);

    // Timed run
    double t_start = MPI_Wtime();
    for (int i = 0; i < ITERATIONS; i++)
        MPI_Alltoall(sendbuf.data(), MSG_SIZE, MPI_DOUBLE,
                     recvbuf.data(), MSG_SIZE, MPI_DOUBLE, MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Alltoall avg time: "
                  << (t_end - t_start) / ITERATIONS
                  << " s\n";

    MPI_Finalize();
    return 0;
}