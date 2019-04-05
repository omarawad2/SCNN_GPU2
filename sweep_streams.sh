#! /bin/bash

./compile.sh
./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>$1/result_streams.rpt

for no_streams in 2 4 8 16 32 64 128 
    do
        sed -i 's/\/\/\#define compute_streams_flag/\#define compute_streams_flag/' main.cu
        sed -i "s/\#define compute_streams [[:digit:]]*/\#define compute_streams $no_streams/" main.cu
        echo -n "$no_streams, ">>$1/result_streams.rpt
        ./compile.sh
        ./SCNN_GPU $1 | sed -n 's/Total time://p'>>$1/result_streams.rpt
    done

