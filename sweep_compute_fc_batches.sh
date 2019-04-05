#! /bin/bash


for num in 128 256 64 
    do
        sed -i "s/\#define compute_fc_batches [[:digit:]]*/\#define compute_fc_batches $num/" main.cu
        echo -n "$num, ">>$1/result_compute_fc_batches.rpt
        ./compile.sh
        ./SCNN_GPU $1 | sed -n 's/Total time://p'>>$1/result_compute_fc_batches.rpt
    done

