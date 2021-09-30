#!/bin/bash

cd dataset
mkdir -p cifar10
wget -P ./cifar10/ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf ./cifar10/cifar-10-python.tar.gz -C ./cifar10/
python3 prepare_cifar10.py --data_dir="./cifar10/cifar-10-batches-py/"
