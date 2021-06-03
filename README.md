# Graph Neural Networks

This repository contains the code support for the Master thesis for the Master in Innovation & Research in Informatics - Data science of BarcelonaTech University, Faculty of Informatics.

The purpose of the repository is two-fold. On the one hand, it provides an implementation of the **gnn** Python module to create *Graph Neural Networks* (GNN) using Tensorflow 2, based on the *Message Passing Neural Network* framework on [[1]](#qm9-ref).

On the other hand it provides two examples of graph neural networks using said **gnn** module as well as the equivalent implementation using the [**iGNNition**](https://ignnition.net/) library, which provides a fast way for prototyping GNN without the need for expert implementations.

The example models are the following:

- Quantum Chemistry QM9 dataset molecules' properties prediction, see [[1]](#qm9-ref).

- Graph Neural Networks for Scalable Radio Resource Management, see [[2]](#radio-ref).

## Environemnt setup

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

Next, we need to install the [**iGNNition**](https://ignnition.net/) library. The repository contains a [*Wheel* file with the 1.0.2 version of the framework](ignnition/ignnition-1.0.2-py3-none-any.whl), to provide reproducibility. To install, activate the environment and run:

```bash
pip install ignnition/ignnition-1.0.2-py3-none-any.whl
```

Alternatively, one could directly install the latest version on PyPi using:

```bash
pip install ignnition
```

>Note that the latest changes in the iGNNition framework may break some of the implementations of this repository, which are tested for version 1.0.2.

## References

1. <a id=qm9-ref ></a>Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. & Dahl, G.E.. (2017). *Neural Message Passing for Quantum Chemistry*. Proceedings of the 34th International Conference on Machine Learning, in Proceedings of Machine Learning Research 70:1263-1272 Available [here](http://proceedings.mlr.press/v70/gilmer17a.html).

2. <a id=radio-ref ></a>Yifei Shen, Yuanming Shi, Jun Zhang, Khaled B. Letaief:
*Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis*. CoRR abs/2007.07632 (2020). Available [here](https://arxiv.org/abs/2007.07632).