#!/usr/bin/env bash

echo "Dataset dir: $1"
python data_farfetch.py --save $1/products.parquet
python data_farfetch.py --save $1/products_train.parquet
python data_farfetch.py --save $1/products_test.parquet

python process_pairs.py --full-size $1