#!/bin/bash
# You can also manually download the data from the google drive https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8
# Then place the downloaded tar.gz file into data/

if [ ! -d gdown.pl ]; then
  git clone https://github.com/circulosmeos/gdown.pl.git
fi

./gdown.pl/gdown.pl 'https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8' data/wmt16_en_de.tar.gz

ZIP=${1:-data/wmt16_en_de.tar.gz}
NUM_WORKERS=${2:-60}

DATA_DIR=data
TEXT_DIR=$DATA_DIR/text/wmt16_en_de
BINARY_DIR=$DATA_DIR/binary/wmt16_en_de

mkdir -p $TEXT_DIR $BINARY_DIR
tar -xzvf $ZIP -C $TEXT_DIR

python preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref $TEXT_DIR/train.tok.clean.bpe.32000 \
  --validpref $TEXT_DIR/newstest2013.tok.bpe.32000 \
  --testpref $TEXT_DIR/newstest2014.tok.bpe.32000 \
  --destdir $BINARY_DIR \
  --nwordssrc 32768 \
  --nwordstgt 32768 \
  --joined-dictionary \
  --workers $NUM_WORKERS
