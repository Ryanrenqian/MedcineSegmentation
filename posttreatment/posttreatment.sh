#!/bin/bash
set -x
model=$1
workspace=$2
dense=$3
tresh=$4
k=$5
densepath=$workspace/dense${dense}
mkdir -p $densepath
python scannet_off_matrix.py -dense $dense -p $model -save $densepath -k $k
csvpath=$densepath/csv
python post_treatment.py -s $csvpath -i $densepath -d $dense
