#!/bin/bash


Path=/coding_linux20/encov_torch/nn_training/src/nn

WeightsPath=/data/user/DATA_SSD1/__adri/weights_multiple_trainings_2

rm -R $WeightsPath/*

batchsize=(4 2 5)
lerningRate=(0.003 0.001 0.0005 0.0001)
optimizer=("sgd" "adam")

compteur=0


for (( batch=0; batch<=2; batch++ ))
do  
    for (( lr=0; lr<=3; lr++ ))
    do
        for (( opt=0; opt<=1; opt++ ))
        do
            python -W ignore -m src.nn.train --config ./src/nn/config/config_endovis.yml -b ${batchsize[$batch]} -l ${lerningRate[$lr]} -n $compteur -o ${optimizer[$opt]}
            
            mkdir $WeightsPath/$compteur

            sleep 2

            echo -e "train $compteur, batchsize : ${batchsize[$batch]}, learning rate : ${lerningRate[$lr]}, optimizer : ${optimizer[$opt]}" >> $WeightsPath/$compteur/train.txt
            
            ((compteur+=1))
        done
    done

done

