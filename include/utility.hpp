#pragma once
#include <iostream>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <omp.h>
#ifndef R123_ASSERT
#include <cassert>
#define R123_ASSERT(x) assert(x)
#endif

#include "include/Random123/philox.h"

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;


extern int world_rank;

template <typename...Args>
auto master_printf(const char* fmt,
		   const Args&...args)
{
  if(world_rank==0)
    printf(fmt,args...);
}

struct Couter
{
  template <typename T>
  Couter& operator<<(T&& t)
  {
    if(world_rank==0)
      std::cout<<t;

    return *this;
  }
};

inline Couter master_cout;

// Creazione di un timer per misurare le prestazioni
struct timer {
    static timer timerCost;

    static auto now() { return high_resolution_clock::now(); }

    size_t tot = 0;
    size_t n = 0;
    high_resolution_clock::time_point from;

    void start() { n++; from = now(); }
    void stop() { tot += duration_cast<nanoseconds>(now()-from).count(); }

    double get(bool sub = true) {
        double res = (double) tot;
        if (sub && timerCost.n > 0)
            res -= timerCost.tot * (double)n / (double)timerCost.n;
        return res / 1e9;
    }
};

// Converte un indice lineare in coordinate multi-dimensionali
inline void index_to_coord(size_t index, int N_dim, const size_t *arr_ptr, size_t *coord_buf) {
    for (int d = 0; d < N_dim; ++d) {
        coord_buf[d] = index % arr_ptr[d];
        index /= arr_ptr[d];
    }
}

// Converte coordinate multi-dimensionali in un indice lineare
inline size_t coord_to_index(int N_dim, const size_t *arr_ptr, const size_t *coord_buf) {
    size_t index = 0;
    size_t mult = 1;
    for (int d = 0; d < N_dim; ++d) {
        index += coord_buf[d] * mult;
        mult *= arr_ptr[d];
    }
    return index;
}

// Calcola l'indice globale di un sito dato il suo indice locale

inline size_t compute_global_index(size_t iSite_local,
                                   const vector<size_t>& local_L,
                                   const vector<size_t>& global_offset,
                                   const vector<size_t>& arr,
                                   int N_dim,
                                   size_t* coord_local,
                                   size_t* coord_global) {
    // Converte l'indice locale in coordinate locali
    index_to_coord(iSite_local, N_dim, local_L.data(), coord_local);

    // Converte coordinate locali in coordinate globali
    for (int d = 0; d < N_dim; ++d) {
        coord_global[d] = coord_local[d] + global_offset[d];
    }

    // Converte coordinate globali in indice globale
    return coord_to_index(N_dim, arr.data(), coord_global);
}

inline uint32_t philox_rand(uint64_t global_idx, uint32_t iConf, uint32_t seed) {
    philox4x32_ctr_t ctr = {{
        (uint32_t)(global_idx),
        (uint32_t)(global_idx >> 32),
        (uint32_t)iConf,
        0
    }};
    philox4x32_key_t key = {{seed, 0}};
    return philox4x32(ctr, key).v[0];
}
