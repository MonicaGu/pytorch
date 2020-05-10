This repository contains the source code implementation of dynamic model parallel training and Pytorch modified to profile runtime statistics for pytorch models.

## Directory Structure

### `model_parallel/runtime`

This contains the Python implementation of runtime for dynamic model parallel training in PyTorch.

### `model_parallel/models`

This contains the partitioned PyTorch models and configuration files on how to allocate the PyTorch models on different machines/GPUs.

## Setup
To run the code, you will need to compile the modified PyTorch code in this repositiry and install PyTorch by running:
```bash
git submodule update --init --recursive
python setup.py install
```

Graphviz is required for the profiler. To install graphviz:
```bash
pip install graphviz
conda install graphviz
```

## End-to-end Workflow
To run a demo, run the following commands.
[from `model_parallel/profiler/image_classification`; you will need to have the changes to PyTorch listed above]
Note that the profiling step must be run with only a single GPU (hence the `CUDA_VISIBLE_DEVICES=0` before the command).

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet152 -b 64 
```

[from `model_parallel/optimizer`]

```bash
python optimizer_graph_hierarchical.py -f ../profiler/image_classification/profiles/resnet152/graph.txt -n 3 --activation_compression_ratio 1 -o resnet152_partitioned
```

[from `model_parallel/optimizer`]

```bash
python convert_graph_to_model.py -f resnet152_partitioned/gpus=4.txt -n ResNet152Partitioned -a vgg16 -o ../runtime/image_classification/models/resnet152/gpus=3 --stage_to_num_ranks 0:1,1:1,2:1
```

[from `model_parallel/`; run on 4 machines (including a master server with CPU and 3 servers with 1 GPU each)]

```bash
python test_socket.py
python test_parallel_0.py --rank 0  --dist_addr <rank0 IP address> 
python test_parallel_0.py --rank 1  --dist_addr <rank0 IP address> 
python test_parallel_0.py --rank 2  --dist_addr <rank0 IP address> 
```

`rank0 IP address` here is the IP address of the rank 0 process (for tirch.distributed).

