#! /bin/bash


for num in 8 16 32 64 128 4 
    do
        sed -i "s/\#define compute_conv_batches [[:digit:]]*/\#define compute_conv_batches $num/" main.cu
        echo -n "$num, ">>result_compute_conv_batches.rpt
        ./compile.sh
        ./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>result_compute_conv_batches.rpt
    done

