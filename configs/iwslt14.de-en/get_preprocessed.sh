#!/bin/bash

mkdir -p data/binary/iwslt14_de_en

wget -O data/binary/iwslt14_de_en/iwslt14_de_en.preprocessed.tgz 'https://www.dropbox.com/s/t5dqiamjdzahhfc/iwslt14_de_en.preproessed.tgz?dl=0'

cd data/binary/iwslt14_de_en

tar -xzvf iwslt14_de_en.preprocessed.tgz
