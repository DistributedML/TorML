#!/bin/bash

for i in `seq 0 5 49`;
do
  for j in `seq 0 5 49`;
  do

    python code/misslabel_dataset.py amazon "$i" "$j"
  done
done
# python code/misslabel_mnist.py
