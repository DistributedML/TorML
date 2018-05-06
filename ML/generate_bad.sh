#!/bin/bash

for i in `seq 0 10`;
do
  for j in `seq 0 10`;
  do
    python code/misslabel_mnist.py "$i" "$j"
  done
done
# python code/misslabel_mnist.py
