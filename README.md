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

The primary objective of the repository is to develop a U-Net model from the paper titled **U-Net: Convolutional Networks for Biomedical Image Segmentation**.

### Test U-Net network

```Bash
python main.py
```

### To do

- Use model list
- Separate Encoder end Decoder into Class
- Block with double conv