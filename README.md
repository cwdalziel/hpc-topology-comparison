# CMSC 714 Research Project

## simgrid install: 

`
sudo apt install libsimgrid-dev
`

## simgrid docs:

https://simgrid.org/doc/latest/index.html

mpi specifically: https://simgrid.org/doc/latest/Tutorial_MPI_Applications.html

## simgrid usage

compile with `smpicxx` or `smpicc` and run with `smpirun` which takes the same args as `mpirun` but with `-platform example.xml` to use network topology defined in xml file

## run_benchmarks example usage (after running make)

`
./run_benchmarks.sh bin/alltoall
`