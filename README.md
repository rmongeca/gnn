# Graph Neural Networks
Graph Neural Network example implementation using Tensorflow 2 API.

This repository also implements some GNN examples such as:
- Quantum Chemistry QM9 molecules' properties prediction

## Set-up

In order to set-up the repository environment, **Python 3.8** and **Virtualenv** are needed.
Refer to you OS specific instructions, for instance in Ubuntu/Debian:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.8 python3.8-dev virtualenv
```

Next, we need to create a virtual environment, *gnn*, to isolate the Python dependencies of the
project:
```bash
virtualenv -p /usr/bin/python3.8 ~/.virtualenvs/gnn
```

Whenever we open a new terminal, we will need to activate this environment with the following
command:
```bash
source ~/.virtualenvs/gnn/bin/activate
```

Finally, we can install the required repository dependencies:
```bash
pip install -r requirements.txt
```