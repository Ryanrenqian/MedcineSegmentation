#!/bin/bash
set -x
model=$1
workspace=$2
dense=$3
tresh=0.4
k=
GPU=1,2,3,4
densepath=$workspace/dense${dense}
csvpath=$densepath/csv
mkdir -p $csvpath $densepath
CUDA_VISIBLE_DEVICESCUDA_VISIBLE_DEVICES=$GPU python posttreatment/scannet_off_matrix.py -dense $dense -pth $model -save $densepath -k $k -thres $thres && \
python posttreatment/post_treatment.py -s $csvpath -i $densepath -d $dense
