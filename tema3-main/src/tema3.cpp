#include <iostream>
#include "computations.h"
#include "topology.h"

/*
Topology matrix model
rank_leader     nr_workers      workers
0:              1               4
1:              2               5, 9
2:              2               6, 7
3:              3               8, 10, 11
*/

int main(int argc, char * argv[]) {
	int rank, num_procs;
    int vec_elems = 0, type = 0;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int** topology = init_topology(num_procs);
    int assigned_coordinator = rank;

    if (argc >= 2 && rank == 0) {
        vec_elems = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        type = std::atoi(argv[2]);
    }
	
    // init coordinators
    if (rank < num_coord) {
        read_data(rank, topology);
    }

    switch (type)
    {
    case 0:
        ring_topology(rank, num_procs, &assigned_coordinator, topology);
        MPI_Barrier(MPI_COMM_WORLD);
        ring_computations(rank, num_procs, assigned_coordinator, vec_elems, topology, -1);
        break;
    
    case 1:
        broken_topology(rank, num_procs, &assigned_coordinator, topology, 1);
        MPI_Barrier(MPI_COMM_WORLD);
        broken_computations(rank, num_procs, assigned_coordinator, vec_elems, topology, -1);
        break;
    
    case 2:
        broken_topology(rank, num_procs, &assigned_coordinator, topology, 2);
        MPI_Barrier(MPI_COMM_WORLD);
        broken_computations(rank, num_procs, assigned_coordinator, vec_elems, topology, PARTITION_RANK_EXCLUDE);
        break;

    default:
        std::cout << "Unknown type, exiting.\n";
        exit(0);
        break;
    }

	MPI_Finalize();
	return 0;
}
