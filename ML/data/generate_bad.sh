#!/bin/bash

for i in `seq 0 9`;
do
  for j in `seq 0 9`;
  do
    python code/misslabel_mnist.py "$i" "$j"
  done
done
# python code/misslabel_mnist.py
