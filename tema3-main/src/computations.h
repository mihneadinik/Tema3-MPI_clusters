#ifndef COMPUTATIONS_H_
#define COMPUTATIONS_H_

#include <iostream>
#include "common.h"
#include <unordered_map>

int count_workers(int** topology) {
    int nr_workers = 0;
    for (int i = 0; i < num_coord; i++) {
        nr_workers += topology[i][0];
    }
    return nr_workers;
}

int find_worker_offset(int rank, int worker_id, int** topology) {
    int offset = 0;
    for (int i = 1; i <= topology[rank][0] && topology[rank][i] != worker_id; i++) {
        offset++;
    }

    return offset;
}

void compute_clusters_info(int** topology, int* start_indexes, int* tasks_per_cluster, double tasks_per_worker, int vec_elems) {
    for (int i = 0; i < num_coord; i++) {
        int num_operations = top_double(tasks_per_worker * topology[i][0]);
        start_indexes[i] = (i > 0) ? start_indexes[i - 1] + tasks_per_cluster[i - 1] : 0;
        if (start_indexes[i] + num_operations > vec_elems) {
            num_operations = vec_elems - start_indexes[i];
        }
        tasks_per_cluster[i] = num_operations;
    }
}

void assemble_cluster_response(int** topology, int* recv_vec, int* vec, int rank, std::unordered_map<int, int>& worker_start_indices, int base_offset = 0) {
    for (int i = 0; i < topology[rank][0]; i++) {
        int total_work, worker_id, worker_offset;
        MPI_Status status;
        MPI_Recv(&total_work, 1, MPI_INT, MPI_ANY_SOURCE, WORKER_TAG, MPI_COMM_WORLD, &status);
        worker_id = status.MPI_SOURCE;
        worker_offset = worker_start_indices[worker_id];
        MPI_Recv(recv_vec, total_work, MPI_INT, worker_id, WORKER_TAG, MPI_COMM_WORLD, &status);

        // copy and clear received computations
        for (int j = 0; j < total_work; j++) {
            vec[j + worker_offset + base_offset] = recv_vec[j];
            recv_vec[j] = 0;
        }
    }
}

void assign_tasks_to_workers(int** topology, int* recv_vec, int rank, double tasks_per_worker, int work_left, std::unordered_map<int, int>& worker_start_indices, int offset = 0) {
    int worker_offset = 0;
    for (int i = 1; i <= topology[rank][0]; i++) {
        int curr_worker = topology[rank][i];
        int start = worker_offset;
        int end = start + top_double(tasks_per_worker);
        int total_work = end - start;

        if (total_work > work_left) {
            total_work = work_left;
        }
        work_left -= total_work;

        send_and_log(&total_work, 1, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);
        send_and_log(&recv_vec[offset + start], total_work, MPI_INT, curr_worker, rank, WORKER_TAG, MPI_COMM_WORLD);

        worker_offset += total_work;
        worker_start_indices[curr_worker] = start;
    }
}

void worker_computations(int rank, int assigned_coordinator, int exclude) {
    // partition exclusion
    if (assigned_coordinator == exclude) {
        return;
    }

    int num_operations;
    MPI_Status status;
    MPI_Recv(&num_operations, 1, MPI_INT, assigned_coordinator, WORKER_TAG, MPI_COMM_WORLD, &status);

    int* recv_vec = (int*) calloc(sizeof(int), num_operations);
    MPI_Recv(recv_vec, num_operations, MPI_INT, assigned_coordinator, WORKER_TAG, MPI_COMM_WORLD, &status);

    // work
    for (int i = 0; i < num_operations; i++) {
        recv_vec[i] *= 5;
    }

    // send back to coordinator (first inform about size)
    send_and_log(&num_operations, 1, MPI_INT, assigned_coordinator, rank, WORKER_TAG, MPI_COMM_WORLD);
    send_and_log(recv_vec, num_operations, MPI_INT, assigned_coordinator, rank, WORKER_TAG, MPI_COMM_WORLD);
}

void ring_computations(int rank, int num_procs, int assigned_coordinator, int vec_elems, int** topology, int exclude) {
    int next = (rank + 1) % num_coord, prev = (rank - 1 + num_coord) % num_coord;
    // leader
    if (rank == 0) {
        std::unordered_map<int, int> worker_start_indices;
        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* vec = create_vec(vec_elems);
        int start_indexes[num_coord] = {0}, tasks_per_cluster[num_coord] = {0};

        // count workers
        int nr_workers = count_workers(topology);
        double tasks_per_worker = (double)vec_elems / nr_workers;
        int offset = 0;

        // compute clusters info
        compute_clusters_info(topology, start_indexes, tasks_per_cluster, tasks_per_worker, vec_elems);

        // send tasks to my cluster (first inform about length, then send vec)
        assign_tasks_to_workers(topology, vec, rank, tasks_per_worker, tasks_per_cluster[rank], worker_start_indices);

        // send tasks to next coordinator (first inform about global and local lengths and offset, then send vec)
        send_and_log(&vec_elems, 1, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(tasks_per_cluster, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(start_indexes, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(vec, vec_elems, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);

        // assemble cluster's response
        assemble_cluster_response(topology, recv_vec, vec, rank, worker_start_indices);

        // assemble coordinator's responses
        MPI_Status status;
        MPI_Recv(recv_vec, vec_elems, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        for (int j = start_indexes[1]; j < vec_elems; j++) {
            vec[j] = recv_vec[j];
            recv_vec[j] = 0;
        }

        print_vec(vec, vec_elems);
    }

    // other coordinators
    if (rank > 0 && rank < num_coord) {
        std::unordered_map<int, int> worker_start_indices;
        int start_indexes[num_coord] = {0}, tasks_per_cluster[num_coord] = {0};
        int offset, worker_offset = 0, num_operations, work_left;
        MPI_Status status;
        MPI_Recv(&vec_elems, 1, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(tasks_per_cluster, num_coord, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(start_indexes, num_coord, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        num_operations = tasks_per_cluster[rank];
        work_left = num_operations;
        offset = start_indexes[rank];
        double tasks_per_worker = (double)num_operations / topology[rank][0];

        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* worker_vec = (int*) calloc(sizeof(int), tasks_per_worker + 1);
        MPI_Recv(recv_vec, vec_elems, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        // assign tasks to workers
        assign_tasks_to_workers(topology, recv_vec, rank, tasks_per_worker, tasks_per_cluster[rank], worker_start_indices, offset);

        // assemble cluster's response
        assemble_cluster_response(topology, worker_vec, recv_vec, rank, worker_start_indices, offset);

        // send to the next coordinator
        if (rank != num_coord - 1) {
            send_and_log(&vec_elems, 1, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(tasks_per_cluster, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(start_indexes, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(recv_vec, vec_elems, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        } else {
            // send final vec to leader
            send_and_log(recv_vec, vec_elems, MPI_INT, 0, rank, RING_TAG, MPI_COMM_WORLD);
        }
    }

    // worker
    if (rank >= num_coord) {
        worker_computations(rank, assigned_coordinator, exclude);
    }
}

void broken_computations(int rank, int num_procs, int assigned_coordinator, int vec_elems, int** topology, int exclude) {
    MPI_Status status;
    int next = (rank - 1 + num_coord) % num_coord, prev = (rank + 1) % num_coord;
    int last = (exclude == -1) ? 1 : 2;
    // broken channel errors
    if (rank == 0) {
        prev = next;
    }
    if (rank == last) {
        next = prev;
    }

    // partition case => coordinator does nothing
    if (rank == exclude) {
        return;
    }

    // leader
    if (rank == 0) {
        std::unordered_map<int, int> worker_start_indices;
        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* vec = create_vec(vec_elems);
        int start_indexes[num_coord] = {0}, tasks_per_cluster[num_coord] = {0};

        // count workers
        int nr_workers = count_workers(topology);
        double tasks_per_worker = (double)vec_elems / nr_workers;
        int offset = 0;

        // compute clusters info
        compute_clusters_info(topology, start_indexes, tasks_per_cluster, tasks_per_worker, vec_elems);

        // send tasks to my cluster (first inform about length, then send vec)
        assign_tasks_to_workers(topology, vec, rank, tasks_per_worker, tasks_per_cluster[rank], worker_start_indices);

        // send tasks to next coordinator (first inform about global and local lengths and offset, then send vec)
        send_and_log(&vec_elems, 1, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(tasks_per_cluster, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(start_indexes, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
        send_and_log(vec, vec_elems, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);

        // assemble cluster's response
        assemble_cluster_response(topology, recv_vec, vec, rank, worker_start_indices);

        // assemble coordinator's responses
        MPI_Recv(recv_vec, vec_elems, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        for (int j = start_indexes[1]; j < vec_elems; j++) {
            vec[j] = recv_vec[j];
            recv_vec[j] = 0;
        }

        print_vec(vec, vec_elems);
    }

    // other coordinators
    if (rank > 0 && rank < num_coord) {
        std::unordered_map<int, int> worker_start_indices;
        int start_indexes[num_coord] = {0}, tasks_per_cluster[num_coord] = {0};
        int offset, worker_offset = 0, num_operations, work_left;
        MPI_Recv(&vec_elems, 1, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(tasks_per_cluster, num_coord, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(start_indexes, num_coord, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        num_operations = tasks_per_cluster[rank];
        work_left = num_operations;
        offset = start_indexes[rank];
        double tasks_per_worker = (double)num_operations / topology[rank][0];

        int* recv_vec = (int*) calloc(sizeof(int), vec_elems);
        int* worker_vec = (int*) calloc(sizeof(int), tasks_per_worker + 1);
        MPI_Recv(recv_vec, vec_elems, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);

        // assign tasks to workers
        assign_tasks_to_workers(topology, recv_vec, rank, tasks_per_worker, tasks_per_cluster[rank], worker_start_indices, offset);

        if (rank != last) {
            // send to the next coordinator
            send_and_log(&vec_elems, 1, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(tasks_per_cluster, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(start_indexes, num_coord, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
            send_and_log(recv_vec, vec_elems, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);

            // receive computed array from neighbour and assemble it
            std::swap(prev, next);
            MPI_Recv(recv_vec, vec_elems, MPI_INT, prev, RING_TAG, MPI_COMM_WORLD, &status);
        }

        assemble_cluster_response(topology, worker_vec, recv_vec, rank, worker_start_indices, offset);

        // send assembled array forward
        send_and_log(recv_vec, vec_elems, MPI_INT, next, rank, RING_TAG, MPI_COMM_WORLD);
    }

    // worker
    if (rank >= num_coord) {
        worker_computations(rank, assigned_coordinator, exclude);
    }
}
#endif