#!/bin/bash
#PBS -N ising_64x64_cold_scan
#PBS -l nodes=1:ppn=32
#PBS -l walltime=08:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

SEED=654321
LOGFILE="logs/example_64x64_cold_${PBS_JOBID}.log"
mkdir -p logs
exec > "$LOGFILE" 2>&1

export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Esempio: 2D Ising 64x64 start hot"
echo "Scan in temperature da T=1.8 a T=3.0"
echo "SEED=$SEED"
echo "Start: $(date)"
echo ""


NDIM=2
NRANKS=1
NTHREADS=8
L0=64
L1=64
NCONFS=10000

mkdir -p output/64x64

echo "Compiling..."
mpicxx -O3 -std=c++17 -fopenmp \
    -Iinclude -Iinclude/include \
    src/main.cpp -o ising_rowing.exe

TEMPS=(1.8 2.0 2.2 2.269 2.4 2.6 2.8 3.0)

for T in "${TEMPS[@]}"; do
    BETA=$(awk "BEGIN {printf \"%.10f\", 1.0/$T}")

    echo "=== T=$T  BETA=$BETA  $(date +%H:%M:%S) ==="

    mpirun -n $NRANKS ./ising_rowing.exe \
        $NDIM $L0 $L1 $NCONFS $NTHREADS $BETA $SEED -cold

    echo " Output: output/64x64/meas_T${T}_hot.txt"
    echo ""
done

echo "Fatto: $(date)"
