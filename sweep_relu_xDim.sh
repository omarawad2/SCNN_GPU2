#! /bin/bash


for num in 2 4 8 16 32 64 512 2048 1024
    do
        sed -i "s/\#define relu_xDim [[:digit:]]*/\#define relu_xDim $num/" main.cu
        echo -n "$num, ">>$1/result_relu_xDim.rpt
        ./compile.sh
        ./SCNN_GPU $1 | sed -n 's/Total time://p'>>$1/result_relu_xDim.rpt
    done

