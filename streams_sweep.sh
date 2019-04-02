#! /bin/bash


for no_streams in 512 1024 2048
    do
        sed -i 's///' main.cu
        echo -n "stream = $no_streams, Execution time = ">>result_streams.rpt
        ./compile.sh
        ./SCNN_GPU.sh bvlc_alexnet | sed -n 's/Total time://p'>>result_streams.rpt
    done

