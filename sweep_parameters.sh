#! /bin/bash

mkdir $1

sweep_bias.sh $1
sweep_bias_yDim.sh $1
sweep_compute_conv_batches.sh $1
sweep_compute_fc_batches.sh $1
sweep_pop_xyDim.sh $1
sweep_streams.sh $1
