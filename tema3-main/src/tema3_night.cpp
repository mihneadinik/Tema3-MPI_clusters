#include <iostream>
#include <fstream>
#include <vector>
#include "mpi.h"

#define num_coord 4
#define topology_sharing_convergence 2
#define RING_TAG 0
#define WORKER_TAG 1

int** init_topology(int num_procs) {
    int** topology = (int**) malloc(sizeof(int*) * num_coord);
    for (int i = 0; i < num_coord; i++) {
        topology[i] = (int*) calloc(sizeof(int), num_procs);
    }

    return topology;
}

void send_and_log(void *buf, int count, MPI_Datatype datatype, int dest, int src, int tag, MPI_Comm comm) {
    MPI_Send(buf, count, datatype, dest, tag, comm);
    std::cout << "M(" << src << "," << dest << ")" << std::endl;
}

void read_data(int rank, int** topology) {
    std::string filename = "cluster" + std::to_string(rank) + ".txt";

    std::ifstream file(filename);
    std::string line;

    // read nr of workers
    std::getline(file, line);
    int nr_workers = std::stoi(line);
    topology[rank][0] = nr_workers;

    // read workers
    for (int i = 1; i <= nr_workers; i++) {
        std::getline(file, line);
        int worker_id = std::stoi(line);
        topology[rank][i] = worker_id;
    }
}

void debug_topology(int rank, int num_procs, int** topology) {
    std::cout << "Proccess " << rank << " printing.\n";
    for (int i = 0; i < num_coord; i++) {
        for (int j = 0; j < num_procs; j++) {
            std::cout << topology[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_topology(int rank, int num_procs, int** topology) {
    std::cout << rank << " -> ";

    for (int i = 0; i < num_coord; i++) {
        std::cout << i << ":";

        // write workers from cluster separated by comma
        int nr_workers = topology[i][0];
        for (int j = 0; j < nr_workers - 1; j++) {
            std::cout << topology[i][j + 1] << ",";
        }

        // write last worker in cluster (if it exists) and add a space afterwards
        if (nr_workers > 0) {
            std::cout << topology[i][nr_workers];
        }

        // don't put a blank space at the end of the line
        if (i != num_coord - 1) {
            std::cout << " ";
        }
    }

    std::cout << std::endl;
}

void ring_communication_share_topologies(int rank, int num_procs, int** topology, int comm_id = 0) {
    int next = (rank + 1) % num_coord, prev = (rank - 1) % num_coord;
    int** recv_topology = init_topology(num_procs);

    // first coordinator initialises the communication
    if (rank == 0) {
        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }

    // receive other coordinator's topology
    for (int i = 0; i < num_coord; i++) {
        MPI_Status status;
        MPI_Recv(recv_topology[i], num_procs, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
    }


    // update curr topology
    for (int i = 0; i < num_coord; i++) {
        if (topology[i][0] == 0) {
            topology[i] = recv_topology[i];
        }
    }

    // after the first cycle, only the last coord knows the whole topology
    // in the second cycle, each coord will get the full topology
    if ((comm_id == 0 && (rank == num_coord - 1 || rank == 0)) || (comm_id == 1 && (rank < num_coord - 1 && rank > 0))) {
        print_topology(rank, num_procs, topology);

        // I have the full topology => send it to my workers
        for (int i = 1; i <= topology[rank][0]; i++) {
            int curr_worker = topology[rank][i];
            for (int j = 0; j < num_coord; j++) {
                send_and_log(topology[j], num_procs, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);
            }
        }
    }

    // close the circle so first coord doesn't loop
    if (rank != 0) {
        // send updated topology in circle
        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }
}

void worker_receive_topology(int rank, int num_procs, int** topology, int* assigned_coordinator) {
    // receive topology from coordinator
    for (int i = 0; i < num_coord; i++) {
        MPI_Status status;
        MPI_Recv(topology[i], num_procs, MPI_INT, MPI_ANY_SOURCE, WORKER_TAG, MPI_COMM_WORLD, &status);
        *assigned_coordinator = status.MPI_SOURCE;
    }

    // print result
    print_topology(rank, num_procs, topology);
}

void type0_topology(int rank, int num_procs, int* assigned_coordinator, int** topology) {
    if (rank >= num_coord) {
        worker_receive_topology(rank, num_procs, topology, assigned_coordinator);
    }

    // after first iteration only coord0 will have the final topology
    // after the second iteration all leaders will have updated topologies
    for (int i = 0; i < topology_sharing_convergence; i++) {
        // coordinators share between them their topology
        if (rank < num_coord) {
            ring_communication_share_topologies(rank, num_procs, topology, i);
        }
    }
}

int* create_vec(int nr_elem) {
    int* vec = (int*) malloc(sizeof(int) * nr_elem);

    for (int i = 0; i < nr_elem; i++) {
        vec[i] = nr_elem - i - 1;
    }

    return vec;
}

int find_worker_offset(int rank, int worker_id, int** topology) {
    int offset = 0;
    for (int i = 1; i <= topology[rank][0] && topology[rank][i] != worker_id; i++) {
        offset++;
    }

    return offset;
}

void print_vec(int* vec, int size) {
    std::cout << "Rezultat: ";

    for (int i = 0; i < size - 1; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << vec[size - 1] << std::endl;
}

void type0_computations(int rank, int num_procs, int assigned_coordinator, int vec_elems, int** topology) {
    // leader
    if (rank == 0) {
        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* vec = create_vec(vec_elems);
        int start_indexes[4] = {0}, tasks_per_cluster[4] = {0};

        // equally assign tasks
        int nr_workers = 0;
        for (int i = 0; i < num_coord; i++) {
            nr_workers += topology[i][0];
        }
        int tasks_per_worker = vec_elems / nr_workers, offset = 0;

        // send tasks to my cluster (first inform about length, then send vec)
        for (int i = 1; i <= topology[rank][0]; i++) {
            int curr_worker = topology[rank][i];
            send_and_log(&tasks_per_worker, 1, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);
            send_and_log(&vec[offset], tasks_per_worker, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);

            offset += tasks_per_worker;
        }

        // compute cluster info
        tasks_per_cluster[0] = tasks_per_worker * topology[0][0];
        for (int i = 1; i < num_coord; i++) {
            int num_operations = tasks_per_worker * topology[i][0];
            start_indexes[i] = offset;
            tasks_per_cluster[i] = num_operations;

            offset += num_operations;
        }

        // send tasks to next coordinator (first inform about global and local lengths and offset, then send vec)
        int num_operations = tasks_per_worker * topology[(rank + 1) % num_coord][0];
        send_and_log(&vec_elems, 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(&num_operations, 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(&start_indexes[(rank + 1) % num_coord], 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(vec, vec_elems, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);

        // assemble cluster's response
        for (int i = 0; i < topology[rank][0]; i++) {
            MPI_Status status;
            MPI_Recv(recv_vec, tasks_per_worker, MPI_INT, MPI_ANY_SOURCE, WORKER_TAG, MPI_COMM_WORLD, &status);

            // copy and clear received computations
            offset = find_worker_offset(rank, status.MPI_SOURCE, topology) * tasks_per_worker;
            for (int j = 0; j < tasks_per_worker; j++) {
                vec[j + offset] = recv_vec[j];
                recv_vec[j] = 0;
            }
        }

        // assemble coordinator's responses
        for (int i = 1; i < num_coord; i++) {
            MPI_Status status;
            MPI_Recv(recv_vec, vec_elems, MPI_INT, MPI_ANY_SOURCE, RING_TAG, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            num_operations = tasks_per_cluster[src];
            offset = start_indexes[src];

            for (int j = 0; j < num_operations; j++) {
                vec[j + offset] = recv_vec[j + offset];
                recv_vec[j + offset] = 0;
            }
        }

        print_vec(vec, vec_elems);
    }

    // coordinator
    if (rank > 0 && rank < num_coord) {
        int num_operations, offset, worker_offset = 0;
        MPI_Status status;
        MPI_Recv(&vec_elems, 1, MPI_INT, (rank - 1) % num_coord, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&num_operations, 1, MPI_INT, (rank - 1) % num_coord, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&offset, 1, MPI_INT, (rank - 1) % num_coord, RING_TAG, MPI_COMM_WORLD, &status);

        int tasks_per_worker = num_operations / topology[rank][0];
        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* worker_vec = (int*) calloc(sizeof(int), tasks_per_worker);
        MPI_Recv(recv_vec, vec_elems, MPI_INT, (rank - 1) % num_coord, RING_TAG, MPI_COMM_WORLD, &status);

        // send to the next coordinator
        if (rank != num_coord - 1) {
            int new_offset = offset + num_operations, new_num_operations = tasks_per_worker * topology[(rank + 1) % num_coord][0];
            send_and_log(&vec_elems, 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(&new_num_operations, 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(&new_offset, 1, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(recv_vec, vec_elems, MPI_INT, (rank + 1) % num_coord, rank, RING_TAG, MPI_COMM_WORLD);
        }

        // assign tasks to workers
        for (int i = 1; i <= topology[rank][0]; i++) {
            int curr_worker = topology[rank][i];
            send_and_log(&tasks_per_worker, 1, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);
            send_and_log(&recv_vec[offset + worker_offset], tasks_per_worker, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);

            worker_offset += tasks_per_worker;
        }

        // assemble cluster's response
        for (int i = 0; i < topology[rank][0]; i++) {
            MPI_Status status;
            MPI_Recv(worker_vec, tasks_per_worker, MPI_INT, MPI_ANY_SOURCE, WORKER_TAG, MPI_COMM_WORLD, &status);

            // copy and clear received computations
            int new_offset = find_worker_offset(rank, status.MPI_SOURCE, topology) * tasks_per_worker;
            for (int j = 0; j < tasks_per_worker; j++) {
                recv_vec[j + offset + new_offset] = worker_vec[j];
                worker_vec[j] = 0;
            }
        }

        // send final vec to leader
        send_and_log(recv_vec, vec_elems, MPI_INT, 0, rank, RING_TAG, MPI_COMM_WORLD);
    }

    // worker
    if (rank >= num_coord) {
        int num_operations;

        MPI_Status status;
        MPI_Recv(&num_operations, 1, MPI_INT, assigned_coordinator, WORKER_TAG, MPI_COMM_WORLD, &status);

        int* recv_vec = (int*) calloc(sizeof(int), num_operations);
        MPI_Recv(recv_vec, num_operations, MPI_INT, assigned_coordinator, WORKER_TAG, MPI_COMM_WORLD, &status);

        // work
        for (int i = 0; i < num_operations; i++) {
            recv_vec[i] *= 5;
        }

        // send back to coordinator
        send_and_log(recv_vec, num_operations, MPI_INT, assigned_coordinator, rank, WORKER_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char * argv[]) {
	int rank, num_procs;
    int vec_elems = 0, type = 0;
    /*
        rank_leader     nr_workers      workers
        0:              1               4
        1:              2               5, 9
        2:              2               6, 7
        3:              3               8, 10, 11
    */

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

    // wait for all leaders to finish inits
    MPI_Barrier(MPI_COMM_WORLD);

    switch (type)
    {
    case 0:
        type0_topology(rank, num_procs, &assigned_coordinator, topology);
        MPI_Barrier(MPI_COMM_WORLD);
        type0_computations(rank, num_procs, assigned_coordinator, vec_elems, topology);
        break;
    
    case 1:
        break;
    
    case 2:
        break;

    default:
        std::cout << "Unknown type, exiting.\n";
        exit(0);
        break;
    }

	MPI_Finalize();
	return 0;
}