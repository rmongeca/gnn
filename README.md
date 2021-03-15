# Graph Neural Networks
Graph Neural Network example implementation using:
 - Custom GNN module using Tensorflow 2.
 - iGNNition UPC-BNN framework -> [homepage](https://ignnition.net/)

This repository also implements some GNN examples such as:
- Quantum Chemistry QM9 molecules' properties prediction

## Set-up

In order to set-up the repository environment, **Python 3.8** and **Conda** are needed.
Refer to you OS specific instructions.

Next, we need to restore the Conda environment, *gnn*, to isolate the Python dependencies of the
project:
```bash
conda env create -f environment.yml
```

Whenever we open a new terminal, we will need to activate this environment with the following
command:
```bash
conda activate gnn
```
