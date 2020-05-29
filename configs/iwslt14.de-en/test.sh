checkpoints_path=$1
configs=$2
metrics=${3:-"normal"}
gpu=${4:-0}
subset=${5:-"test"}

output_path=$(dirname -- "$checkpoints_path")
out_name=$(basename -- "$configs")

mkdir -p $output_path/exp

CUDA_VISIBLE_DEVICES=$gpu python generate.py \
        --data data/binary/iwslt14_de_en  \
        --path "$checkpoints_path" \
        --gen-subset $subset \
        --beam 5 \
        --batch-size 128 \
        --remove-bpe \
        --configs=$configs \
        > $output_path/exp/${out_name}_${subset}_gen.out

GEN=$output_path/exp/${out_name}_${subset}_gen.out

SYS=$GEN.sys
REF=$GEN.ref

# get normal BLEU or SacreBLEU score
if [ $metrics = "normal" ]
then
  echo "Evaluate Normal BLEU score!"
  grep ^H $GEN | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS
  grep ^T $GEN | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF
  python score.py --sys $SYS --ref $REF
elif [ $metrics = "sacre" ]
then
  echo "Evaluate SacreBLEU score!"
  grep ^H $GEN | cut -f3- > $SYS.pre
  grep ^T $GEN | cut -f2- > $REF.pre
  sacremoses detokenize < $SYS.pre > $SYS
  sacremoses detokenize < $REF.pre > $REF
  python score.py --sys $SYS --ref $REF --sacrebleu
fi

