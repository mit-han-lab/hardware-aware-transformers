checkpoints_path=$1
avg_checkpoints=${2:-10}

model=average_model_$avg_checkpoints.pt
output_path=$checkpoints_path

python average_checkpoints.py \
  --inputs $output_path \
  --num-epoch-checkpoints $avg_checkpoints \
  --output $output_path/$model
