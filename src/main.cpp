#include "utility.hpp"
#include "ising.hpp"
#include "metropolis.hpp"
#include "halo.hpp"
#include "io.hpp"
#include <cstdint>
#include <vector>
#include <cstdio>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mpi.h>

using namespace std;

int world_rank; // Rank (id) del processo
int world_size; // Numero di rank

long long N=1;       // Numero totale di siti
double Beta;         // Inverso della temperatura
int nThreads;        // Numero di thread OpenMP
int N_dim;           // Numero di dimensioni
vector<size_t> global_L;  // Lunghezze del reticolo globale per dimensione
int nConfs;          // Numero di configurazioni
size_t seed;         // Seed per il generatore di numeri casuali

// Definizione della variabile statica timerCost (dichiarata in utility.hpp)
timer timer::timerCost;

int main(int argc, char** argv) {

    timer totalTime, computeTime, mpiTime, ioTime, setupTime, bulkTime;
    totalTime.start();
    setupTime.start();

    // Sottrae il costo del timer alla simulazione
    for (size_t i = 0; i < 100000; ++i) { 
        timer::timerCost.start(); 
        timer::timerCost.stop(); 
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      master_printf("world_size  %d rank %d\n", world_size, world_rank);
    
    // Controlla il flag cold: serve per avere 2 simulazioni, una con conf
    // iniziale random e un' altra dove tutti gli spin sono allineati.
    bool cold_start = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-cold") == 0) {
            cold_start = true;
            // Rimuovi il flag dagli argomenti
            for (int j = i; j < argc - 1; ++j)
                argv[j] = argv[j + 1];
            argc--;
            break;
        }
    }

    if (world_rank == 0) {
        if (argc > 1) {
            N_dim = atoi(argv[1]);
            int expected = N_dim + 6;
            if (argc != expected) {
                fprintf(stderr, "Uso: %s <N_dim> <L0> [L1 ...] <nConfs> <nThreads> <Beta> <seed> [-cold]\n", argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
            global_L.resize(N_dim);
            for (int d = 0; d < N_dim; ++d)
                global_L[d] = (size_t)atoll(argv[2 + d]);
            nConfs = atoi(argv[2 + N_dim]);
            nThreads = atoi(argv[3 + N_dim]);
            Beta = atof(argv[4 + N_dim]);
            seed = (size_t)atoll(argv[5 + N_dim]);
        } else {
            fprintf(stderr, "Uso: %s <N_dim> <L0> [L1 ...] <nConfs> <nThreads> <Beta> <seed> [-cold]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    //Broadcast dati agli altri processi
    MPI_Bcast(&N_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        global_L.resize(N_dim);
    }
    MPI_Bcast(global_L.data(), N_dim*sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nConfs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nThreads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&seed, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    { int cs = cold_start; MPI_Bcast(&cs, 1, MPI_INT, 0, MPI_COMM_WORLD); cold_start = cs; }

    omp_set_num_threads((int)nThreads);
    
    vector<int> Chunks(N_dim); //Numero di processi allocati lungo ogni dimensione
    vector<size_t> local_L(N_dim); //Dimensioni locali del nodo
    vector<int> rank_coords(N_dim); // Coordinate cartesiane del rank nella griglia MPI
    MPI_Comm cart_comm;              // Comunicatore cartesiano
    vector<int> periods(N_dim, 1);  // Condizioni periodiche ai bordi
    
    MPI_Dims_create(world_size, N_dim, Chunks.data());
    MPI_Cart_create(MPI_COMM_WORLD,(int)N_dim,Chunks.data(),
                    periods.data(),1,&cart_comm);
    MPI_Cart_coords(cart_comm, world_rank, N_dim, 
                    rank_coords.data());

    //Vettore che contiene, per ogni sito al confine, il rank dei 
    //processi MPI dei suoi vicini lungo ogni dimensione.
    std::vector<std::vector<int>> neighbors;
    halo_index(cart_comm, N_dim, neighbors);

    size_t N_local = 1; // Numero di siti del nodo
    size_t N_alloc = 1; // Numero di siti del nodo + halo
    vector<size_t> local_L_halo(N_dim); //Lato del rank + halo
    vector<size_t> global_offset(N_dim); //Offset per passare da coordinate 
                                         //del rank a coordinate globali

    // Si controlla che il reticolo sia divisibile in Chunks uguali
    for (int d = 0; d < N_dim; ++d) {
        if (global_L[d] % Chunks[d] != 0) {
            if (world_rank == 0){
                cerr << "Errore: arr[" << d << "] non divisibile per" 
                <<"Chunks uguali. Prova un'altra combinazione di rank"
                <<"e lati[" << d << "]\n";
                }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        local_L[d] = global_L[d] / Chunks[d];
        local_L_halo[d] = local_L[d] + 2;
        N_local *= local_L[d];
        N_alloc *= local_L_halo[d];
        N *= global_L[d];
        global_offset[d] = rank_coords[d] * local_L[d]; 
    }

    //Precalcola gli stride locali a ogni rank
    vector<uint32_t> stride_halo(N_dim);
    stride_halo[0] = 1;
    for (int d = 1; d < N_dim; ++d){
        stride_halo[d] = stride_halo[d-1] * local_L_halo[d-1];
    }
    //Precalcola gli stride globali
    vector<long long> stride_global(N_dim);
    stride_global[0] = 1;
    for (int d = 1; d < N_dim; ++d){
        stride_global[d] = stride_global[d-1] * global_L[d-1];
    }

    // Vettori per siti al confine (Red/Black), usati da entrambi i path
    vector<uint32_t> boundary_sites[2];
    vector<uint32_t> boundary_indices[2];
    
    // Inizializzazione RNG Philox counter-based: il seed seleziona l'esperimento,
    // il counter (global_idx, iConf) seleziona il numero random
    uint32_t rng_seed = (uint32_t)(seed + 104729);
    print_simulation_info(N_dim, N, nThreads, nConfs, Beta,
                          sizeof(uint32_t));
    // Vettore che contiene la configurazione locale a ogni rank
    // (1 byte per sito). Contiene celle halo.
    vector<int8_t> conf_local(N_alloc); 

    //Genera la prima configurazione
    initialize_configuration(conf_local, N_local, N_dim, local_L,
                            local_L_halo, global_offset, global_L, rng_seed,
                            cold_start);

    // Costruisce le dimensioni delle facce e la sua posizione
    vector<FaceInfo> faces = build_faces(local_L, N_dim);
    // Salva gli indici (locali e globali) dei siti che appartengono
    // a ogni faccia, distinti per parità globale.
    vector<FaceCache> face_cache = build_face_cache(faces, local_L,
                                                    local_L_halo,
                                                    global_offset,
                                                    global_L, N_dim,
                                                    boundary_sites,
                                                    boundary_indices);

    vector<MPI_Request> requests; //definizione richieste processi MPI
    HaloBuffers buffers; //definizione buffers
    buffers.resize(N_dim);

    //Halo exchange per calcolo energia e magnetizzazione iniziale
    start_halo_exchange(conf_local, local_L, local_L_halo,
                        neighbors, cart_comm, N_dim, buffers,
                        requests, face_cache, 0);
    write_halo_data(conf_local, buffers, local_L,
                    local_L_halo, N_dim, face_cache, 0, requests);
    start_halo_exchange(conf_local, local_L, local_L_halo,
                        neighbors, cart_comm, N_dim, buffers,
                        requests, face_cache, 1);
    write_halo_data(conf_local, buffers, local_L,
                    local_L_halo, N_dim, face_cache, 1, requests);
                    
    long long E_rank = computeEn_rank(conf_local, stride_halo, 
                                        local_L, N_dim);
    long long Mag_rank = compute_Mag_rank(conf_local, stride_halo, 
                                            local_L, N_dim);

    // Riduzione per calcolo di energia e magnetizzazione iniziale (globale)
    long long E = 0;
    long long Mag = 0;
    MPI_Reduce(&E_rank, &E, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&Mag_rank, &Mag, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);

    // Lookup table per precalcolare l'esponenziale usata in Metropolis
    // Le differenze di energia positive non 0 possibili sono N_dim
    // e sono multipli di 4, dato che le energie possibili sono multipli
    // di 2 e eDiff=-2*E (se flippa). Esempi di differenze di energia 
    // 2D=[4,8] 3D=[4,8,12]

    vector<double> expTable(N_dim);
    for (int d = 0; d < N_dim; ++d){
        expTable[d] = exp(-Beta * 4.0 * (d + 1));
    }

    // Apertura del file di output per le misure
    // Directory: output/{L0}x{L1}x.../  File: meas_T{T}_{hot|cold}.txt
    FILE* measFile = nullptr;
    if (world_rank == 0) {
        double T = 1.0 / Beta;

        // Crea la directory output/{L0}x{L1}x...
        string dir = "output/";
        for (int d = 0; d < N_dim; ++d)
            dir += (d == 0 ? "" : "x") + to_string(global_L[d]);
        string mkdir_cmd = "mkdir -p " + dir;
        system(mkdir_cmd.c_str());

        char Tbuf[32];
        snprintf(Tbuf, sizeof(Tbuf), "%.6g", T);
        string fname = dir + "/meas_T" + string(Tbuf)
                     + (cold_start ? "_cold" : "_hot") + ".txt";

        measFile = fopen(fname.c_str(), "w");
        if (!measFile) {
            perror(("Errore apertura " + fname).c_str());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        master_printf("Output file: %s\n", fname.c_str());
    }

    setupTime.stop();

    for (int iConf = 0; iConf < nConfs; ++iConf) {

        // Aggiornamento Rosso/Nero.
        // L'ordine dei processi é il seguente:
        //1)avvia la comunicazione MPI per gli halo neri
        //2)update Bulk rosso
        //3)aspetta la sincronizzazione e scrive gli halo neri
        //4)update Boundary rosso
        //5)avvia la comunicazione MPI per gli halo rossi
        //6)update Bulk nero
        //7)aspetta la sincronizzazione e scrive gli halo rossi
        //8)update Boundary nero
        
        // Variazione di energia e magnetizzazione locali a ogni rank
        long long DeltaE = 0;
        long long DeltaMag = 0;

        for(int updPar=0;updPar<2;updPar++){

            const int commPar=1-updPar;
            // Inizia l' halo exchange nero/rosso
            mpiTime.start();
            start_halo_exchange(conf_local, local_L,
                                local_L_halo, neighbors,
                                cart_comm, N_dim, buffers,
                                requests,
                                face_cache,commPar);
            mpiTime.stop();

            computeTime.start();
            bulkTime.start();
            // Update Bulk rosso/nero (include divisioni per righe)
            metropolis_update_bulk(conf_local,updPar,
                                    local_L, local_L_halo,
                                    global_offset, global_L,
                                    stride_halo, stride_global, expTable,
                                    DeltaE, DeltaMag,
                                    rng_seed, iConf);
            bulkTime.stop();
            computeTime.stop();
            mpiTime.start();

            // Scrivi gli halo
            write_halo_data(conf_local, buffers,
                            local_L, local_L_halo, N_dim,
                            face_cache, commPar, requests);
            mpiTime.stop();
            computeTime.start();
            // Update boundary rossa/nero
            metropolis_update_border(conf_local, boundary_sites[updPar],
                                     boundary_indices[updPar], stride_halo,
                                     expTable, DeltaE,DeltaMag, rng_seed,
                                     iConf, (size_t)nThreads);
            computeTime.stop();
        } //Fine loop sulle paritá
            

        // Somma delle variazioni locali a ogni rank per ottenere
        // le variazioni di energia e magnetizzazione globali
        mpiTime.start();
        long long DeltaE_glob = 0;
        long long DeltaMag_glob = 0;
        MPI_Reduce(&DeltaE, &DeltaE_glob, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
        MPI_Reduce(&DeltaMag, &DeltaMag_glob, 1, MPI_LONG_LONG, MPI_SUM, 0, cart_comm);
        mpiTime.stop();
        
        // Si scrivono le misure nel file
        if (world_rank == 0) {
            ioTime.start();
            E += DeltaE_glob;
            Mag += DeltaMag_glob;

            double e = (double)E / (double)N;
            double m = (double)Mag / (double)N;
            fprintf(measFile, "%lg %lg\n", e, m);
            if ((iConf + 1) % 1000 == 0)
                fflush(measFile);

            ioTime.stop();
        }
    } // Fine del loop sulle configurazioni

    if (world_rank == 0 && measFile) {
        fclose(measFile);
    }

    totalTime.stop();
    
    if (world_rank == 0) {
        print_performance_summary(totalTime.get(), computeTime.get(),
                                  mpiTime.get(), ioTime.get(),
                                  setupTime.get(), nConfs);
        master_printf("Bulk (rowing) time: %.6f s (%.6f s/conf) | "
                      "Boundary time: %.6f s (%.6f s/conf)\n",
                      bulkTime.get(), bulkTime.get() / nConfs,
                      computeTime.get() - bulkTime.get(),
                      (computeTime.get() - bulkTime.get()) / nConfs);
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
