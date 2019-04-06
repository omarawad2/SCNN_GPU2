#!/bin/bash

cd SCNN_GPU
filename='./../cp_commits.log'

while read line; do
	git stash
    git checkout $line
    if grep -q 'std::vector<Layer> read_vgg_cnn_s()' main.cu;
    	then
    		sed 's///' main.cu
    	else
			sed -i 's/bvlc_alexnet/vgg_cnn_s/g' main.cu
			sed -i 's/true,4,0/true,2,0/g' main.cu
			sed -i 's/true,1,2/true,1,0/g' main.cu
			sed -i 's/uint64_t getMaxIndex/uint64_t Layer::getMaxIndex/g' gpu_Layer.cu
	fi
	#cp ./../gpu_Layer.cu .
	#cp ./../compile.sh .
	sed -i 's/out_act_shape\[0\]/1/g' Layer.cpp		
	sed -i 's/act_shape\[0\]/1/g' Layer.cpp
	sed -i 's/out_act_shape\[0\]/1/g' gpu_Layer.cu	
	sed -i 's/act_shape\[0\]/1/g' gpu_Layer.cu
	sed -i 's/= read_bvlc_alexnet()/= read_vgg_cnn_s()/g' main.cu
    ./compile.sh
    echo -n "$line," >>./../$1/cp_result.rpt
    ./SCNN_GPU $1 | sed -n 's/Total time://p'>>./../$1/cp_result.rpt
done < $filename
