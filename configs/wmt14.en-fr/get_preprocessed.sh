#!/bin/bash

mkdir -p data/binary/wmt14_en_fr

wget -O data/binary/wmt14_en_fr/wmt14_en_fr.preprocessed.tgz 'https://www.dropbox.com/s/mrs8efjrrnc61xi/wmt14_en_fr.preprocessed.tgz?dl=0'

cd data/binary/wmt14_en_fr

tar -xzvf wmt14_en_fr.preprocessed.tgz
