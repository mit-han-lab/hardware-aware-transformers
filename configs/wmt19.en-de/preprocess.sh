#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
if [ ! -d mosesdecoder ]; then
    git clone https://github.com/moses-smt/mosesdecoder.git
fi

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
if [ ! -d subword-nmt ]; then
    git clone https://github.com/rsennrich/subword-nmt.git
fi

NUM_WORKERS=${1:-60}

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

URLS=(
    "http://www.statmt.org/europarl/v9/training/europarl-v9.de-en.tsv.gz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release3/en-de.bicleaner07.txt.gz"
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz"
    "http://data.statmt.org/wikititles/v1/wikititles-v1.de-en.tsv.gz"
    "https://s3-eu-west-1.amazonaws.com/tilde-model/rapid2019.de-en.zip"
    "http://data.statmt.org/wmt19/translation-task/dev.tgz"
    "http://data.statmt.org/wmt19/translation-task/test.tgz"
)
FILES=(
    "europarl-v9.de-en.tsv.gz"
    "en-de.bicleaner07.txt.gz"
    "training-parallel-commoncrawl.tgz"
    "news-commentary-v14.de-en.tsv.gz"
    "wikititles-v1.de-en.tsv.gz"
    "rapid2019.de-en.zip"
    "dev.tgz"
    "test.tgz"
)
SPLIT=(
    "en-de.bicleaner07.txt"
    "news-commentary-v14.de-en.tsv"
    "wikititles-v1.de-en.tsv"
    "europarl-v9.de-en.tsv"
)
CORPORA=(
    "en-de.bicleaner07.txt"
    "news-commentary-v14.de-en.tsv"
    "wikititles-v1.de-en.tsv"
    "europarl-v9.de-en.tsv"
    "commoncrawl.de-en"
    "rapid2019.de-en"
)


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi


OUTDIR=data/text/wmt19_en_de
src=en
tgt=de
lang=en-de
prep=$OUTDIR/tokenized
tmp=$prep/tmp
orig=$OUTDIR/orig

mkdir -p $orig $tmp $prep

cd $orig


for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip -k $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done

cd -

cut -f1 $orig/en-de.bicleaner07.txt > $orig/en-de.bicleaner07.txt.en
cut -f2 $orig/en-de.bicleaner07.txt > $orig/en-de.bicleaner07.txt.de
cut -f1 $orig/news-commentary-v14.de-en.tsv > $orig/news-commentary-v14.de-en.tsv.de
cut -f2 $orig/news-commentary-v14.de-en.tsv > $orig/news-commentary-v14.de-en.tsv.en
cut -f1 $orig/wikititles-v1.de-en.tsv > $orig/wikititles-v1.de-en.tsv.de
cut -f2 $orig/wikititles-v1.de-en.tsv > $orig/wikititles-v1.de-en.tsv.en
cut -f1 $orig/europarl-v9.de-en.tsv > $orig/europarl-v9.de-en.tsv.de
cut -f2 $orig/europarl-v9.de-en.tsv > $orig/europarl-v9.de-en.tsv.en

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l -f
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads $NUM_WORKERS -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done


echo "pre-processing dev data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2017-ende-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads $NUM_WORKERS -a -l $l > $tmp/valid.$l
    echo ""
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/dev/newstest2018-ende-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads $NUM_WORKERS -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    cp $tmp/train.tags.$lang.tok.$l $tmp/train.$l
done

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done


prep=data/text/wmt19_en_de/tokenized
TEXT_DIR=$prep
BINARY_DIR=data/binary/wmt19_en_de

# use the same dict to get same preprocessed data
wget -O data/text/wmt19_en_de/wmt19ende_dict.txt 'https://www.dropbox.com/s/u9h8ns6xe8ux91d/wmt19ende_dict.txt?dl=0'

python preprocess.py \
    --source-lang en \
    --target-lang de \
    --trainpref $TEXT_DIR/train \
    --validpref $TEXT_DIR/valid \
    --testpref $TEXT_DIR/test \
    --destdir $BINARY_DIR \
    --thresholdtgt 1 \
    --thresholdsrc 1 \
    --joined-dictionary \
    --workers $NUM_WORKERS \
    --tgtdict data/text/wmt14_en_fr/wmt19ende_dict.txt \
