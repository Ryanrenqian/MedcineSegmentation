#!/bin/bash
set -x
model=$1
workspace=$2
dense=$3
tresh=$4
k=$5
GPU=$6
densepath=$workspace/dense${dense}
csvpath=$densepath/csv
CUDACUDA_VISIBLE_DEVICES=$GPU python posttreatment/scannet_off_matrix.py -dense $dense -p $model \
 -save $densepath -k $k -thres $thres && \
python posttreatment/post_treatment.py -s $csvpath -i $densepath -d $dense
