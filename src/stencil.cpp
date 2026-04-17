#include <mpi.h>

#include <iostream>
#include <vector>

constexpr int GRID_SIZE = 256;
constexpr int ITERATIONS = 100;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Arrange ranks in 2D cartesian grid
    int dims[2] = {0, 0};
    int periods[2] = {1, 1};  // wraparound = torus-like logical layout
    MPI_Dims_create(size, 2, dims);

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int left, right, up, down;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);

    int local_n = GRID_SIZE / dims[0];
    // +2 for halo rows/cols on each side
    std::vector<double> grid((local_n + 2) * (local_n + 2), 0.0);

    double t_start = MPI_Wtime();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Exchange halo rows (left/right neighbors)
        MPI_Sendrecv(grid.data() + local_n, local_n, MPI_DOUBLE, left, 0,
                     grid.data() + 0, local_n, MPI_DOUBLE, right, 0,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(grid.data() + local_n * local_n, local_n, MPI_DOUBLE, right, 1,
                     grid.data() + local_n * (local_n + 1), local_n, MPI_DOUBLE, left, 1,
                     cart_comm, MPI_STATUS_IGNORE);

        // Exchange halo cols (up/down neighbors)
        MPI_Sendrecv(grid.data() + local_n, local_n, MPI_DOUBLE, up, 2,
                     grid.data() + 0, local_n, MPI_DOUBLE, down, 2,
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(grid.data() + local_n * local_n, local_n, MPI_DOUBLE, down, 3,
                     grid.data() + local_n * (local_n + 1), local_n, MPI_DOUBLE, up, 3,
                     cart_comm, MPI_STATUS_IGNORE);

        // Stencil update (5-point)
        for (int i = 1; i <= local_n; i++)
            for (int j = 1; j <= local_n; j++)
                grid[i * (local_n + 2) + j] = 0.25 * (grid[(i - 1) * (local_n + 2) + j] +
                                                      grid[(i + 1) * (local_n + 2) + j] +
                                                      grid[i * (local_n + 2) + (j - 1)] +
                                                      grid[i * (local_n + 2) + (j + 1)]);
    }
    double t_end = MPI_Wtime();

    if (rank == 0)
        std::cout << "Stencil avg time per iter: "
                  << (t_end - t_start) / ITERATIONS
                  << " s\n";

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}