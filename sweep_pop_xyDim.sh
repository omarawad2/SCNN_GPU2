#! /bin/bash


for num in 1 2  8 16 32 64 4
    do
        sed -i "s/\#define populate_xyDim [[:digit:]]*/\#define populate_xyDim $num/" main.cu
        echo -n "$num, ">>result_populate_xyDim.rpt
        ./compile.sh
        ./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>result_populate_xyDim.rpt
    done

