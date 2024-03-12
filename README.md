# pytorch-UNet
Simple implementation of U-Net network for multi-class classification

### Download and Install Miniconda

```Bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

### Create environment

```Bash
conda create --name unet python==3.9
conda activate unet
```

### Install dependencies

```Bash
pip install -r requirements.txt 
```

### Goal of the repository

The main goal of the repository is just to develop from scratch a U-Net model to a train a classifier based on the Pytorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

The code from the the tutorial remains but the network is different. 