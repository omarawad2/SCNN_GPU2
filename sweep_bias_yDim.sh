#! /bin/bash


for num in 2 4 8 32 64 16
    do
        sed -i "s/\#define bias_yDim [[:digit:]]*/\#define bias_yDim $num/" main.cu
        echo -n "$num, ">>result_bias_yDim.rpt
        ./compile.sh
        ./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>result_bias_yDim.rpt
    done

