#ifndef TOPOLOGY_H_
#define TOPOLOGY_H_

#include <iostream>
#include "common.h"

int** init_topology(int num_procs) {
    int** topology = (int**) malloc(sizeof(int*) * num_coord);
    for (int i = 0; i < num_coord; i++) {
        topology[i] = (int*) calloc(sizeof(int), num_procs);
    }

    return topology;
}

void print_topology(int rank, int num_procs, int** topology) {
    std::cout << rank << " -> ";

    for (int i = 0; i < num_coord; i++) {
        if (topology[i][0]) {
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
    }

    std::cout << std::endl;
}

void topology_update(int** topology, int** recv_topology) {
    for (int i = 0; i < num_coord; i++) {
        if (topology[i][0] == 0) {
            topology[i] = recv_topology[i];
        }
    }
}

void send_topology_to_workers(int** topology, int rank, int num_procs) {
    for (int i = 1; i <= topology[rank][0]; i++) {
            int curr_worker = topology[rank][i];
            for (int j = 0; j < num_coord; j++) {
                send_and_log(topology[j], num_procs, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);
            }
        }
}

void worker_receive_topology(int rank, int num_procs, int** topology, int* assigned_coordinator) {
    // wait to receive topology from coordinator
    for (int i = 0; i < num_coord; i++) {
        MPI_Status status;
        MPI_Recv(topology[i], num_procs, MPI_INT, MPI_ANY_SOURCE, WORKER_TAG, MPI_COMM_WORLD, &status);
        *assigned_coordinator = status.MPI_SOURCE;
    }

    // print result (after receiving it completely)
    print_topology(rank, num_procs, topology);
}

void broken_ring_communication_share_topologies(int rank, int num_procs, int** topology, int type) {
    int next = (rank - 1 + num_coord) % num_coord, prev = (rank + 1) % num_coord;
    int last = (type == 1) ? 1 : 2;
    // broken channel errors
    if (rank == 0) {
        prev = next;
    }
    if (rank == last) {
        next = prev;
    }

    int** recv_topology = init_topology(num_procs);

    // first coordinator initialises the communication and waits to receive it back
    if (rank == 0) {
        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }

        for (int i = 0; i < num_coord; i++) {
            MPI_Status status;
            MPI_Recv(recv_topology[i], num_procs, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        }

        // update curr topology
        topology_update(topology, recv_topology);

        // now it has the full topology => print and send to workers
        print_topology(rank, num_procs, topology);
        send_topology_to_workers(topology, rank, num_procs);
    }

    // last coordinator breaks and turns around the circle
    if ((type == 1 && rank == 1) || (type == 2 && rank == 2)) {
        for (int i = 0; i < num_coord; i++) {
            MPI_Status status;
            MPI_Recv(recv_topology[i], num_procs, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        }

        // update curr topology
        topology_update(topology, recv_topology);

        // now it has the full topology => print and send to workers
        print_topology(rank, num_procs, topology);
        send_topology_to_workers(topology, rank, num_procs);

        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }

    // the other coordinators have to do double work inside the communication channel
    if (rank != 0 && rank != last) {
        // partition case
        if (type == 2 && rank == 1) {
            // now it has the full topology => print and send to workers
            print_topology(rank, num_procs, topology);
            send_topology_to_workers(topology, rank, num_procs);
            return;
        }

        // receive and store updating topology, then forward it
        for (int i = 0; i < num_coord; i++) {
            MPI_Status status;
            MPI_Recv(recv_topology[i], num_procs, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        }

        // update curr topology
        topology_update(topology, recv_topology);

        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }

        // receive and store final topology, then forward it (reverse order in comm)
        std::swap(prev, next);
        for (int i = 0; i < num_coord; i++) {
            MPI_Status status;
            MPI_Recv(recv_topology[i], num_procs, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        }

        // update curr topology
        topology_update(topology, recv_topology);

        // now it has the full topology => print and send to workers
        print_topology(rank, num_procs, topology);
        send_topology_to_workers(topology, rank, num_procs);

        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }
}

void ring_communication_share_topologies(int rank, int num_procs, int** topology, int comm_id = 0) {
    int next = (rank + 1) % num_coord, prev = (rank - 1 + num_coord) % num_coord;
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
    topology_update(topology, recv_topology);

    // after the first cycle, only the last coord knows the whole topology
    // in the second cycle, each coord will get the full topology
    if ((comm_id == 0 && (rank == num_coord - 1 || rank == 0)) || (comm_id == 1 && (rank < num_coord - 1 && rank > 0))) {
        print_topology(rank, num_procs, topology);

        // I have the full topology => send it to my workers
        send_topology_to_workers(topology, rank, num_procs);
    }

    // close the circle so first coord doesn't loop
    if (rank != 0) {
        // send updated topology in circle
        for (int i = 0; i < num_coord; i++) {
            send_and_log(topology[i], num_procs, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }
}

void broken_topology(int rank, int num_procs, int* assigned_coordinator, int** topology, int type) {
    if (rank >= num_coord) {
        worker_receive_topology(rank, num_procs, topology, assigned_coordinator);
    } else {
        broken_ring_communication_share_topologies(rank, num_procs, topology, type);
    }
}

void ring_topology(int rank, int num_procs, int* assigned_coordinator, int** topology) {
    if (rank >= num_coord) {
        worker_receive_topology(rank, num_procs, topology, assigned_coordinator);
    } else {
        for (int i = 0; i < topology_sharing_convergence; i++) {
            // coordinators share between them their topology
            if (rank < num_coord) {
                ring_communication_share_topologies(rank, num_procs, topology, i);
            }
        }
    }
}
#endif