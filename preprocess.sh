#!/usr/bin/env bash

if [ -n "$1" ] && [ ! -d "$1" ]
then
  echo "$1 is not a valid directory"
  exit
elif [ -n "$1" ]
then
  Dir=$1
  echo "Dataset directory: $Dir"
else
  echo "Usage: $0 dataset-directory"
  exit
fi

python data_farfetch.py --save $Dir/products.parquet
python process_pairs.py $Dir
python process_pairs.py --full-size $Dir

echo "Done processing"
