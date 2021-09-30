# Scalable Low-Rank Compression for Neural Networks 

This repository contains the Tensorflow implementation of [our paper](https://doi.org/10.24963/ijcai.2021/447).
The implementation can reproduce the result on CIFAR-10.

If you find our work useful in your research, please consider citing:
```
@inproceedings{ijcai2021-447,
  title     = {Decomposable-Net: Scalable Low-Rank Compression for Neural Networks},
  author    = {Yaguchi, Atsushi and Suzuki, Taiji and Nitta, Shuhei and Sakata, Yukinobu and Tanizawa, Akiyuki},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {3249--3256},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/447},
  url       = {https://doi.org/10.24963/ijcai.2021/447},
}
```

# Environment
- 1 GPU with more than 8GB RAM (recommended)  
- Ubuntu 18.04 LTS  
- Anaconda 4.7.12 (or newer)  

# Usage
1. Create and activate _conda env_:
```shell
conda create -n tf1-py3 tensorflow-gpu=1.15.0 anaconda
conda activate tf1-py3
```

2. Prepare the dataset:
```shell
sh prepare_cifar10.sh
```

3. Run training:
```shell
sh run_cifar10.sh {gpu_id} ./dataset/cifar10/train.txt ./dataset/cifar10/test.txt vgg15 0.01 0.25 KyKxCin_Cout 0
```