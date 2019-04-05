#! /bin/bash

filename='./cp_commits.log'

while read line; do
    git checkout $line
    ./compile.sh
    echo -n "$line," >>result.rpt
    ./SCNN_GPU $1 | sed -n 's/Total time://p'>>$1/cp_result.rpt
done < $filename

