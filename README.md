This repository contains the source code implementation of Pytorch modified to profile runtime statistics for pytorch models.

## Setup
You will need to compile the modified PyTorch code in this repositiry. Note that torchvision should be installed before compiling PyTorch if you want to train a model from torchvision:
```bash
pip install torchvision
git submodule update --init --recursive
python setup.py install
```
