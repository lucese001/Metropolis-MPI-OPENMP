#!/bin/bash
#PBS -N ising_repro_test
#PBS -l nodes=1:ppn=8
#PBS -l walltime=01:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

# Setup MVAPICH2
source /storage/local/exp_soft/local_al9/mpi/mvapich2-2.3.7-2/install/bin/mpivars.sh

SEED=42
LOGFILE="logs/repro_test_${PBS_JOBID}.log"
mkdir -p logs
exec > "$LOGFILE" 2>&1

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Test di riproducibilità:"
echo "SEED=$SEED"
echo "Data: $(date)"
echo ""


NDIM=2
L0=16
L1=16
NCONFS=1000
BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/2.269}")

mkdir -p output/16x16

mpicxx -O3 -std=c++17 -fopenmp \
    -Iinclude -Iinclude/include \
    src/main.cpp -o ising_test.exe

TEST_CONFIGS=("1 1" "1 2" "1 4" "2 2" "2 4")

for config in "${TEST_CONFIGS[@]}"; do
    read NRANKS NTHREADS <<< "$config"
    echo ""
    echo "Config: NRANKS=$NRANKS, NTHREADS=$NTHREADS"
    echo ""

    RUN1_FILE="run1_nr${NRANKS}_nt${NTHREADS}.txt"
    RUN2_FILE="run2_nr${NRANKS}_nt${NTHREADS}.txt"

    echo "  RUN 1"
    mpiexec -n $NRANKS ./ising_test.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED -cold > "$RUN1_FILE" 2>&1

    echo "  RUN 2 "
    mpiexec -n $NRANKS ./ising_test.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED -cold > "$RUN2_FILE" 2>&1

    if diff "$RUN1_FILE" "$RUN2_FILE" > /dev/null 2>&1; then
        echo " Output identici"
    else
        echo " Output diversi"
        echo "  Differenze:"
        diff "$RUN1_FILE" "$RUN2_FILE" | head -10
    fi

    rm -f "$RUN1_FILE" "$RUN2_FILE"
    echo ""
done

echo "Completato: $(date)"