#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <fstream>
#include "mpi.h"

#define num_coord 4
#define topology_sharing_convergence 2
#define PARTITION_RANK_EXCLUDE 1
#define RING_TAG 0
#define WORKER_TAG 1

int min(double a, double b) {
    return (a < b) ? a : b;
}

int top_double(double a) {
    return ((int)a == a) ? a : (int)a + 1;
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

int* create_vec(int nr_elem) {
    int* vec = (int*) malloc(sizeof(int) * nr_elem);

    for (int i = 0; i < nr_elem; i++) {
        vec[i] = nr_elem - i - 1;
    }

    return vec;
}

void print_vec(int* vec, int size) {
    std::cout << "Rezultat: ";

    for (int i = 0; i < size - 1; i++) {
        std::cout << vec[i] << " ";
    }
    std::cout << vec[size - 1] << std::endl;
}
#endif