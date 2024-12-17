#!/bin/bash

# バッチサイズのリストを設定
batch_sizes=(16 32 64 128 256)

# 結果保存用ディレクトリ
output_dir="/data1/yamazono/setRetrieval/DeepFurniture/output/models"

# エポック数とpatienceの設定
epochs=200
patience=10

# ループでバッチサイズごとに実行
for batch_size in "${batch_sizes[@]}"; do
    echo "Running with batch_size=$batch_size"
    python run.py --output_dir "$output_dir" --epochs "$epochs" --batch_size "$batch_size" --patience "$patience"
done

echo "All runs completed."
