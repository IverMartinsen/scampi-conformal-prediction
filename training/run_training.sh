#!/bin/bash

for seed in {1..10}; do
  time_stamp=$(date +%Y%m%d%H%M%S)
  data_dir="./data/BaileyLabels/imagefolder-bisaccate"
  output_dir="./postprocessing/trained_models/${time_stamp}_seed${seed}"
  mkdir -p $output_dir

  python postprocessing/train_classifier.py \
    --batch_size 256 \
    --lr 1e-3 \
    --pretrained_weights "checkpoint.pth" \
    --epochs 100 \
    --data_dir "$data_dir" \
    --weight_decay 1.0 \
    --backbone_arch "vit_small" \
    --output_dir $output_dir \
    --validation_split 0.2 \
    --seed $seed > "$output_dir/train_output.txt"

  python postprocessing/eval_classifier.py \
    --device "cuda" \
    --backbone_weights "checkpoint.pth" \
    --classifier_weights "$output_dir/classifier.pth" \
    --data_dir "$data_dir" \
    --output_dir $output_dir \
    --seed $seed \
    --validation_split 0.2 > "$output_dir/eval_output.txt"
done