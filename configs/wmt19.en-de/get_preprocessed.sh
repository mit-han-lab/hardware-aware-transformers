#!/bin/bash

mkdir -p data/binary/wmt19_en_de

wget -O data/binary/wmt19_en_de/wmt19_en_de.preprocessed.tgz 'https://www.dropbox.com/s/q2st4ox2na9z5z2/wmt19_en_de.preprocessed.tgz?dl=0'

cd data/binary/wmt19_en_de

tar -xzvf wmt19_en_de.preprocessed.tgz
