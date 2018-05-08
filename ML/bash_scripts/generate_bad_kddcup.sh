#!/bin/bash

for i in `seq 0 22`;
do
  for j in `seq 0 22`;
  do
    python code/misslabel_dataset.py kddcup "$i" "$j"
  done
done
# python code/misslabel_mnist.py
