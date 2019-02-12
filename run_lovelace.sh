#!/bin/bash

HOME="/LUSTRE/users/atorres/"
d=$(date +%Y%m%d_%H%M)

mkdir -p $d

for width in 60; do #30 60 90 180 360; do
    #$HOME/pywrap.sh ./src/train_model.py $HOME/data/oahu_min.feather $((width*24*60))
    qsub -pe omp 10 -N "model_60" $HOME/pywrap.sh ./src/train_model.py $HOME/data/oahu_min.feather $((width))
done
