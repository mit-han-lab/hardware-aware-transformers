#!/bin/bash

mkdir -p data/binary/wmt16_en_de

wget -O data/binary/wmt16_en_de/wmt16_en_de.preprocessed.tgz 'https://www.dropbox.com/s/axfwl1vawper8yk/wmt16_en_de.preprocessed.tgz?dl=0'

cd data/binary/wmt16_en_de

tar -xzvf wmt16_en_de.preprocessed.tgz
