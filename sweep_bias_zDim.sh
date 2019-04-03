#! /bin/bash


for num in 1 2  8 16 4
    do
        sed -i "s/\#define bias_zDim [[:digit:]]*/\#define bias_zDim $num/" main.cu
        echo -n "$num, ">>result_bias_zDim.rpt
        ./compile.sh
        ./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>result_bias_zDim.rpt
    done

