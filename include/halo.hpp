#pragma once
#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mpi.h>
#include "utility.hpp"

using std::vector;

// Tipo per passare i buffer per riferimento
struct HaloBuffers {
    vector<vector<int8_t>> send_minus, send_plus; 
    vector<vector<int8_t>> recv_minus, recv_plus; 
    
    void resize(int N_dim) {
        send_minus.resize(N_dim);  
        send_plus.resize(N_dim);
        recv_minus.resize(N_dim);
        recv_plus.resize(N_dim);
    }
};

// Tipo per passare per riferimento le informazioni delle facce
struct FaceInfo {
    vector<size_t> dims;
    vector<size_t> map;
};

struct FaceCache {
    size_t face_size;
    //nota: ogni vettore è composto da due vettori,
    // uno per il caso pari e uno per il caso dispari.
    std::array<vector<size_t>, 2> idx_minus;      // Indici del boundary negativo (per fare Send)
    std::array<vector<size_t>, 2> idx_plus;       // Indici del boundary positivo (per fare Send)
    std::array<vector<size_t>, 2> idx_halo_minus; // Indici della regione halo negativa (per fare Recv/Write)
    std::array<vector<size_t>, 2> idx_halo_plus;  // Indici della regione halo positiva (per fare Recv/Write)
};

// Costruisce le informazioni delle facce per lo scambio halo
inline vector<FaceInfo> build_faces(const vector<size_t>& local_L, 
                                    int N_dim) {

    vector<FaceInfo> faces(N_dim);
    for (int d = 0; d < N_dim; ++d) {
        for (int k = 0; k < N_dim; ++k) {
            if (k == d) continue;
            faces[d].dims.push_back(local_L[k]);
            faces[d].map.push_back(k);
        }
    }
    return faces;
}

// NOTA SULLA TERMINOLOGIA "CACHE": non si tratta di CPU L1/L2/L3 cache.
// È una struttura dati che precomputa e immagazzina gli indici dei siti
// boundary per ogni dimensione e parità. Una volta pre-computati,
// questi indici vengono utilizzati durante ogni scambio halo e update
// senza doverli ricalcolare.
inline vector<FaceCache>
build_face_cache(const vector<FaceInfo>& faces,
                 const vector<size_t>& local_L,
                 const vector<size_t>& local_L_halo,
                 const vector<size_t>& global_offset,
                 const vector<size_t>& arr,
                 int N_dim,
                 vector<uint32_t> boundary_sites[2],
                 vector<uint32_t> boundary_indices[2])
{
    // Alloca la cache (una per dimensione)
    vector<FaceCache> cache(N_dim);

    // Stride globali per calcolare global_idx dei siti boundary
    // idx_global = x*stride_global[0] + y*stride_global[1] + ...
    vector<size_t> stride_global(N_dim);
    stride_global[0] = 1;
    for (int d = 1; d < N_dim; ++d)
        stride_global[d] = stride_global[d-1] * arr[d-1];

    // seen[halo_idx] traccia quali siti di boundary sono già stati inseriti.
    // Necessario per evitare duplicati agli spigoli che compaiono come
    // boundary in più dimensioni contemporaneamente.
    // Esempio: in 3D (0,0,0) è boundary nelle dimensioni 0, 1 e 2.
    size_t N_alloc = 1;
    for (int d = 0; d < N_dim; ++d) N_alloc *= local_L_halo[d];
    vector<bool> seen(N_alloc, false);

    // Per ogni dimensione, calcoliamo gli indici dei siti
    // sulla faccia positiva e negativa.
    for (int d = 0; d < N_dim; ++d) {

        const vector<size_t>& face_dims    = faces[d].dims;
        const vector<size_t>& face_to_full = faces[d].map;
        
        //Calcola il numero di siti della faccia
        size_t face_size = 1;
        for (size_t i = 0; i < face_dims.size(); ++i)
            face_size *= face_dims[i];

        cache[d].face_size = face_size;

        // Pre-riserva spazio nella face cache, la
        // metà dei siti totali +1 per sicurezza
        for (int p = 0; p < 2; ++p) {
            cache[d].idx_minus [p].reserve(face_size / 2 + 1);
            cache[d].idx_plus [p].reserve(face_size / 2 + 1);
            cache[d].idx_halo_minus[p].reserve(face_size / 2 + 1);
            cache[d].idx_halo_plus [p].reserve(face_size / 2 + 1);
        }

        // Itera su tutti i siti della faccia d (seriale)
        vector<size_t> coord_face(face_dims.size());
        vector<size_t> coord_full(N_dim);

        for (size_t i = 0; i < face_size; ++i) {
            // i = indice lineare sulla faccia (0 .. face_size-1)
            index_to_coord(i, (int)face_dims.size(), face_dims.data(), coord_face.data());

            // Calcola i contributi al calcolo delle parità globale
            // e l'indice locale per le dimensioni diverse da d.
            size_t base = 0; //somma delle coordinate globali (senza d)
            size_t base_global = 0; //indice globale (senza d)
            for (size_t j = 0; j < face_to_full.size(); ++j) {
                int k  = (int)face_to_full[j];
                coord_full[k] = coord_face[j] + 1;
                size_t gc = coord_face[j] + global_offset[k]; 
                base += gc;
                base_global += gc * stride_global[k];
            }

            // Calcola la parità globale del sito sulla faccia d
            // Ogni sito sulla faccia d ha 2 lati: negativo (x=0) e positivo (x=L[d]-1)
            // Per l'halo: x=-1 (neg_halo) e x=L[d] (pos_halo)
            int par_pos_face = (int)((base + global_offset[d] + local_L[d] - 1) % 2);
            int par_pos_face_halo = (int)((base + global_offset[d] + local_L[d] ) % 2);
            int par_neg_face = (int)((base + global_offset[d] ) % 2);
            int par_neg_face_halo = (int)((base + global_offset[d] + 1 ) % 2);

            // faccia meno: coord_halo[d] = 1 (local coord[d] = 0)
            coord_full[d] = 1;
            size_t idx_inner_minus = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());
            cache[d].idx_minus[par_neg_face].push_back(idx_inner_minus);

            // global_coord[d] = 0 + global_offset[d]
            uint32_t h_neg = (uint32_t)idx_inner_minus;
            if (!seen[h_neg]) {
                seen[h_neg] = true;
                boundary_sites[par_neg_face].push_back(h_neg);
                boundary_indices[par_neg_face].push_back((uint32_t)(base_global + global_offset[d]
                    * stride_global[d]));
            }

            // halo meno: coord_halo[d] = 0
            coord_full[d] = 0;
            cache[d].idx_halo_minus[par_neg_face_halo].push_back(
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));

            // faccia più: coord_halo[d] = local_L[d] (local coord[d] = local_L[d]-1)
            coord_full[d] = local_L[d];
            size_t idx_inner_plus = coord_to_index(N_dim, local_L_halo.data(), coord_full.data());
            cache[d].idx_plus[par_pos_face].push_back(idx_inner_plus);

            // global_coord[d] = local_L[d]-1 + global_offset[d]
            uint32_t h_pos = (uint32_t)idx_inner_plus;
            if (!seen[h_pos]) {
                seen[h_pos] = true;
                boundary_sites[par_pos_face].push_back(h_pos);
                boundary_indices[par_pos_face].push_back((uint32_t)(base_global + (local_L[d]
                    - 1 + global_offset[d]) * stride_global[d]));
            }

            // halo più: coord_halo[d] = local_L[d]+1
            coord_full[d] = local_L[d] + 1;
            cache[d].idx_halo_plus[par_pos_face_halo].push_back(
                coord_to_index(N_dim, local_L_halo.data(), coord_full.data()));
        }

    } // fine loop su d

    return cache;
}

// Inizia lo scambio halo non-blocking
inline void start_halo_exchange(
    vector<int8_t>& conf_local,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    const vector<vector<int>>& neighbors,
    MPI_Comm cart_comm,
    int N_dim,
    HaloBuffers& buffers,
    vector<MPI_Request>& requests,
    const vector<FaceCache>& cache,
    int parity)
{
    requests.clear();

    buffers.send_minus.resize(N_dim);
    buffers.send_plus.resize(N_dim);
    buffers.recv_minus.resize(N_dim);
    buffers.recv_plus.resize(N_dim);

    for (int d = 0; d < N_dim; ++d) {

        // Calcola la dimensione della faccia (tiene conto della parità)
        const size_t send_minus_size = cache[d].idx_minus[parity].size();
        const size_t send_plus_size  = cache[d].idx_plus[parity].size();
        const size_t recv_minus_size = cache[d].idx_halo_minus[parity].size();
        const size_t recv_plus_size  = cache[d].idx_halo_plus[parity].size();

        // Alloca i buffer
        buffers.send_minus[d].resize(send_minus_size);
        buffers.send_plus[d].resize(send_plus_size);
        buffers.recv_minus[d].resize(recv_minus_size);
        buffers.recv_plus[d].resize(recv_plus_size);

        // Prepara i buffer con le configurazioni
        // (solo siti della parità richiesta)
        for (size_t i = 0; i < send_minus_size; ++i) {
            buffers.send_minus[d][i] = conf_local[cache[d].idx_minus[parity][i]];
        }

        for (size_t i = 0; i < send_plus_size; ++i) {
            buffers.send_plus[d][i] = conf_local[cache[d].idx_plus[parity][i]];
        }

        int tag_minus = 100 + d;
        int tag_plus  = 200 + d;

        MPI_Request req;

        // Ricevi da vicino "dietro"
        MPI_Irecv(buffers.recv_minus[d].data(),
                  recv_minus_size, MPI_INT8_T,
                  neighbors[d][0], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);

        // Ricevi da vicino "davanti"
        MPI_Irecv(buffers.recv_plus[d].data(),
                  recv_plus_size, MPI_INT8_T,
                  neighbors[d][1], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "dietro"
        MPI_Isend(buffers.send_minus[d].data(),
                  send_minus_size, MPI_INT8_T,
                  neighbors[d][0], tag_minus,
                  cart_comm, &req);
        requests.push_back(req);

        // Invia a vicino "davanti"
        MPI_Isend(buffers.send_plus[d].data(),
                  send_plus_size, MPI_INT8_T,
                  neighbors[d][1], tag_plus,
                  cart_comm, &req);
        requests.push_back(req);
    }
}


// Scrive i dati ricevuti nelle regioni halo
inline void write_halo_data(
    vector<int8_t>& conf_local,
    const HaloBuffers& buffers,
    const vector<size_t>& local_L,
    const vector<size_t>& local_L_halo,
    int N_dim,
    const vector<FaceCache>& cache,
    int parity,
    vector<MPI_Request>& requests)
{
    // Aspetta la finalizzazione della comunicazione MPI
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();

    for (int d = 0; d < N_dim; ++d) {

        // Determina la dimensione della faccia (considerando la parità)
        const size_t halo_minus_size = cache[d].idx_halo_minus[parity].size();
        const size_t halo_plus_size  = cache[d].idx_halo_plus[parity].size();

        for (size_t i = 0; i < halo_minus_size; ++i) {

            // Scrive i dati ricevuti negli halo meno
            conf_local[cache[d].idx_halo_minus[parity][i]] = buffers.recv_minus[d][i];
        }
        for (size_t i = 0; i < halo_plus_size; ++i) {

            // Scrive i dati ricevuti negli halo più
            conf_local[cache[d].idx_halo_plus[parity][i]] = buffers.recv_plus[d][i];
        }
    }
}

// Calcola gli indici dei vicini usando la topologia cartesiana MPI
inline void halo_index(MPI_Comm cart_comm, int N_dim,
                      vector<vector<int>>& neighbors) {
    neighbors.resize(N_dim);
    for (int d = 0; d < N_dim; ++d) {
        neighbors[d].resize(2);
        MPI_Cart_shift(cart_comm, d, 1, &neighbors[d][0], &neighbors[d][1]);
    }
}

