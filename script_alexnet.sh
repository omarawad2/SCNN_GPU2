#! /bin/bash

cd /nfs/ug/homes-4/e/elgamma8/CUDA/SCNN/SCNN_GPU/
filename='./cp_commits.log'

while read line; do
    git checkout $line
    ./compile.sh
    echo -n "$line," >>result.rpt
    ./SCNN_GPU bvlc_alexnet | sed -n 's/Total time://p'>>result.rpt
done < $filename

