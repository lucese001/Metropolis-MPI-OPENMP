Introduzione




Questo repositorio include una simulazione di un modello di Ising n dimensionale con algoritmo di Metropolis. Per favorire la performance é stato utilizzato un approccio ibrido, ovvero una parallelizzazione ibrida OPEN MP+MPI. Il file simula un numero di configurazioni e stampa l'energia e la magnetizzazione del reticolo per ogni configurazione.

Directories





src é dove si trova il main.


include é dove si trovano le funzioni chiamate nel main.


utility.hpp contiene la dichiarazione di un timer, la funzione che converte indici in coordinate, quella che converte coordinate in indici e quella che trasforma gli indici locali del nodo in indici globali (necessari per l'rng). Infine c'é pure una funzione che classifica siti e coordinate dei siti in interne (non richiedono halo) ed esterne (richiedono halo).


halo.hpp contiene diverse funzioni utili per lo scambio degli halo, ovvero le celle contigue che appartengono a siti MPI diversi.


ising.hpp contiene la funzione che calcola l'energia di un sito, la magnetizzazione e l'energia di un nodo e la funzione che genera la prima configurazione del reticolo.


io.hpp contiene le funzioni usate per leggere il file di input e stampare diversi dati sulla performance del programma.


metropolis.hpp contiene l'algoritmo di Metropolis che regola l'aggiornamento dei siti.


prng_engine.hpp contiene il generatore di numeri casuali.




input contiene il file dimensioni.txt, che ha gli input necessari per lanciare il programma. Primo riga: numero dimensioni. Seconda riga: dimensioni lati del reticolo (deve essere coerente col numero di dimensioni). Terza riga: numero di configurazioni. Quarta riga: numero threads OPEN MP. Quinta riga: beta (inverso temperatura).






output contine il file meas.txt, che contiene energia e magnetizzazione del reticolo a ogni configurazione.




Compilazione:

mpic++ -O3 -std=c++17 -fopenmp   src/main.cpp   -Iinclude   -o ising




Esecuzione:

mpirun -n <numero processi> ./ising
