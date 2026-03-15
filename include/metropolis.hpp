#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "utility.hpp"
#include "ising.hpp"

// Variabili globali esterne (definite in main.cpp)
extern int N_dim;
extern long long N;
extern double Beta;

// metropolis_update_border: esegue uno sweep Metropolis dei siti al
// bordo.

void metropolis_update_border(vector<int8_t>& conf_local,
                              const vector<uint32_t>& sites,
                              const vector<uint32_t>& sites_global_indices,
                              const vector<uint32_t>& stride_halo,
                              const vector<double>& expTable,
                              long long &DeltaE,
                              long long &DeltaMag,
                              uint32_t rng_seed,
                              int iConf,
                              size_t nThreads)
{
        long long local_dE = 0; // Variazione di energia locale a ogni thread
        long long local_dM = 0; // Variazione di magnetizzazione locale a ogni thread
#pragma omp parallel reduction(+:local_dE, local_dM)

    {
        const size_t iThread = omp_get_thread_num();
        const size_t chunkSize = (sites.size() + nThreads - 1) / nThreads;
        const size_t beg = chunkSize * iThread;
        const size_t end = std::min(sites.size(), beg + chunkSize);

        for (size_t idx = beg; idx < end; ++idx) {
            const uint32_t iSite_halo = sites[idx];
            const uint32_t global_idx = sites_global_indices[idx];

            const int8_t oldVal = conf_local[iSite_halo];
            
            const int8_t proposed_spin = -oldVal;

            const int enBefore = computeEnSite(conf_local, iSite_halo,
                                               stride_halo, N_dim);
            const int eDiff = -2 * enBefore;

            // Accettazione
            bool accept;
            if (eDiff <= 0) {
                accept = true;
            } else {
                // eDiff > 0: valori possibili 4, 8, ..., 4*N_dim
                // Lookup: expTable[eDiff/4 - 1] = exp(-Beta*eDiff)
                uint32_t rand1 = philox_rand(global_idx, iConf, rng_seed);
                const double rand_uniform = (double)rand1 / 4294967296.0;
                accept = (rand_uniform < expTable[eDiff / 4 - 1]);
            }

            if (accept) {
                conf_local[iSite_halo] = proposed_spin;
                local_dE += eDiff;
                local_dM += proposed_spin - oldVal;
            }
        }
    }
    // Somma delle energie per thread
    DeltaE += local_dE;
    DeltaMag += local_dM;
}


// metropolis_update_bulk: aggiorna i siti bulk iterando sulle
// coordinate, dove ogni coordinata va da 1 a local_L[d]-2.
// Ci si restringe a un iperpiano di dimensioni N_dim-1: in pratica,
// ogni riga è una combinazione fissa di (x[1], ..., x[N_dim-1]) .
// Da lí, si aggiorna tutto il bulk facendo partire righe lungo tutte
// le direzioni ortogonali all'iperpiano. Per ogni colonna, il loop
// interno itera su x[0] con passo 2 (scacchiera).

void metropolis_update_bulk(
    vector<int8_t>& conf_local,
    int parity,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<size_t>& global_offset,
    const vector<size_t>& arr,
    const vector<uint32_t>& stride_halo,
    const vector<long long>& stride_global,
    const vector<double>& expTable,
    long long& DeltaE, long long& DeltaMag,
    uint32_t rng_seed, int iConf)  {

        // Calcolo numero di righe
        size_t n_rows = 1;
        for (int d = 1; d < N_dim; d++){
            n_rows *= (local_L[d] - 2);
        }

        long long local_dE = 0, local_dM = 0;

        // for sulle righe
        #pragma omp parallel for schedule(static) reduction(+:local_dE, local_dM)
        for (size_t row = 0; row < n_rows; row++) {

            size_t base_halo   = 0;  // Contributo per l'indice halo
            size_t base_global = 0;  // Contributo per l' indice globale 
            size_t parity_sum  = 0;  // Contributo per la paritá globale
            size_t tmp = row;
            for (int d = 1; d < N_dim; d++) {
                size_t range_d = local_L[d] - 2; // Quante coordinate valide
                size_t x_d = 1 + tmp % range_d;    // Coordinata locale in [1, local_L[d]-2]
                tmp /= range_d;
                //+ 1 perché l'halo aggiunge +1 in ogni dimensione
                base_halo   += (x_d + 1) * stride_halo[d];
                base_global += (x_d + global_offset[d]) * stride_global[d];
                parity_sum  += x_d + global_offset[d];
            }

            // Determina x[0] di partenza correto con la paritá globale
            size_t x0 = 1;
            if (((x0 + global_offset[0] + parity_sum) % 2) != (size_t)parity){
               x0 = 2; 
            }

            // Se x0 è fuori dal bulk, niente siti su questa riga
            if (x0 + 2 > local_L[0]) continue;
   
            // Calcolo indici di partenza sulla riga
            size_t halo_idx   = base_halo   + (x0 + 1);
            size_t global_idx = base_global + (x0 + global_offset[0]);

            // Loop sulla riga dove avanza di 2 per mantenere la parità
            for (size_t x = x0; x + 2 <= local_L[0]; x += 2) {

                const int8_t oldVal = conf_local[halo_idx];
                const int8_t proposed_spin = -oldVal;

                const int enBefore = computeEnSite(conf_local, halo_idx,
                                                stride_halo, N_dim);
                const int eDiff = -2 * enBefore;

                bool accept;
                if (eDiff <= 0) {
                    accept = true;
                } else {
                    uint32_t rand1 = philox_rand(global_idx, iConf, rng_seed);
                    double rand_uniform = (double)rand1 / 4294967296.0;
                    accept = (rand_uniform < expTable[eDiff / 4 - 1]);
                }

                if (accept) {
                    conf_local[halo_idx] = proposed_spin;
                    local_dE += eDiff;
                    local_dM += proposed_spin - oldVal;
                }

                halo_idx   += 2;   
                global_idx += 2;   
            }
        }

        DeltaE += local_dE;
        DeltaMag += local_dM;
}

