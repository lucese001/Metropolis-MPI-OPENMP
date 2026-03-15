#pragma once
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <mpi.h>
#include "utility.hpp"

using std::vector;


// Stampa il riepilogo delle prestazioni alla fine della simulazione
inline void print_performance_summary(double total, double compute,
                                       double mpi, double io, double init,
                                       int nConfs) {

    double overhead = total - compute - mpi - io - init;
    master_printf("\n");
    master_printf("          PRESTAZIONI              \n");
    master_printf("Tempo totale:             %10.3f s (100.0%%)\n", total);
    master_printf("Update Metropolis:          %10.3f s (%5.1f%%)\n", compute, 100.0*compute/total);
    master_printf("Comunicazione MPI:         %10.3f s (%5.1f%%)\n", mpi, 100.0*mpi/total);
    master_printf("I/O:          %10.3f s (%5.1f%%)\n", io, 100.0*io/total);
    master_printf("Tempo di inizializzazione:       %10.3f s (%5.1f%%)\n", init, 100.0*init/total);
    master_printf("Overhead:                  %10.3f s (%5.1f%%)\n", overhead, 100.0*overhead/total);
    master_printf("Configurazioni:               %d\n", nConfs);
    master_printf("Tempo per configurazione:           %10.3f s\n", total/nConfs);
}

// Stampa le informazioni sulla simulazione
inline void print_simulation_info(int N_dim, long long N, size_t nThreads, int nConfs,
                                   double Beta, size_t rng_memory) {
    master_printf("N_dim: %d, Npunti: %zu, NThreads: %d, nConfs: %d, Beta: %lg\n",
           N_dim, N, nThreads, nConfs, Beta);
    master_printf("Uso di memoria dell' rng: %zu Bytes\n", rng_memory);
}


