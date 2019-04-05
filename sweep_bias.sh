#! /bin/bash


for num in  8 32 16
    do
        num2=`expr 1024 / $num / $num`
        sed -i "s/\#define bias_xDim [[:digit:]]*/\#define bias_xDim $num/" main.cu
        sed -i "s/\#define bias_yDim [[:digit:]]*/\#define bias_yDim $num/" main.cu
        sed -i "s/\#define bias_zDim [[:digit:]]*/\#define bias_zDim $num2/" main.cu
        echo -n "$num, $num, $num2, ">>$1/result_bias.rpt
        ./compile.sh
        ./SCNN_GPU $1 | sed -n 's/Total time://p'>>$1/result_bias.rpt
    done

