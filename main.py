from torch.nn import functional as F
import torch
from torch import nn

def double_conv_block(in_channels, out_channels, kernel_size, stride):
    """
    Double convolutional block consisting of two convolutional layers with ReLU activation.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - kernel_size: Size of the convolutional kernel.
    - stride: Stride value for the convolution.

    Returns:
    - conv: Sequential container with two Conv2d layers and ReLU activation.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride),
        nn.ReLU())
    return conv


def single_conv_block(in_channels, out_channels, kernel_size, stride):
    """
    Single convolutional block consisting of one convolutional layer with ReLU activation.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - kernel_size: Size of the convolutional kernel.
    - stride: Stride value for the convolution.

    Returns:
    - conv: Sequential container with one Conv2d layer and ReLU activation.
    """
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.ReLU())
    return conv


def trans_conv_block(in_channels, out_channels, kernel_size, stride):
    """
    Transpose convolutional block consisting of one transpose convolutional layer with ReLU activation.

    Parameters:
    - in_channels: Number of input channels.
    - out_channels: Number of output channels.
    - kernel_size: Size of the transpose convolutional kernel.
    - stride: Stride value for the transpose convolution.

    Returns:
    - trans_conv: Sequential container with one ConvTranspose2d layer and ReLU activation.
    """
    trans_conv = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.ReLU(inplace=True))
    return trans_conv


def crop_tensor(input_tensor, target_tensor):
    """
    Crop the input tensor to match the size of the target tensor.

    Parameters:
    - input_tensor: Input tensor to be cropped.
    - target_tensor: Target tensor with the desired size.

    Returns:
    - cropped_tensor: Cropped tensor with the same size as the target tensor.
    """
    target_size = target_tensor.size()[2]
    tensor_size = input_tensor.size()[2]
    delta = (tensor_size - target_size) // 2

    return input_tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class Unet(nn.Module):
    """
    U-Net architecture for semantic segmentation.

    The U-Net consists of an encoder, a bottleneck, and a decoder.

    Attributes:
    - conv1 to conv9: Double convolutional blocks for encoding and decoding.
    - pool1 to pool4: Max-pooling layers for downsampling in the encoder.
    - tconv1 to tconv4: Transpose convolutional blocks for upsampling in the decoder.
    - out: Single convolutional block for the final output.

    Methods:
    - forward(x): Forward pass through the U-Net.

    """
    def __init__(self):
        super().__init__()

        # Encoder  
        self.conv1 = double_conv_block(in_channels=3, out_channels=64,kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = double_conv_block(in_channels=64, out_channels=128,kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = double_conv_block(in_channels=128, out_channels=256,kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = double_conv_block(in_channels=256, out_channels=512,kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.conv5 = double_conv_block(in_channels=512, out_channels=1024,kernel_size=3, stride=1)
        # Decoder
        self.tconv1 = trans_conv_block(in_channels=1024, out_channels=512,kernel_size=2, stride=2)
        self.conv6 = double_conv_block(in_channels=1024, out_channels=512,kernel_size=3, stride=1)
        self.tconv2 = trans_conv_block(in_channels=512, out_channels=256,kernel_size=2, stride=2)
        self.conv7 = double_conv_block(in_channels=512, out_channels=256,kernel_size=3, stride=1)
        self.tconv3 = trans_conv_block(in_channels=256, out_channels=128,kernel_size=2, stride=2)
        self.conv8 = double_conv_block(in_channels=256, out_channels=128,kernel_size=3, stride=1)
        self.tconv4 = trans_conv_block(in_channels=128, out_channels=64,kernel_size=2, stride=2)
        self.conv9 = double_conv_block(in_channels=128, out_channels=64,kernel_size=3, stride=1)
        self.out = single_conv_block(in_channels=64, out_channels=2,kernel_size=1, stride=1)

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Parameters:
        - x: Input tensor.

        Returns:
        - out: Output tensor after passing through the U-Net.
        """

        # Encoder
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x3 = self.conv2(x2)
        x4 = self.pool2(x3)
        x5 = self.conv3(x4)
        x6 = self.pool3(x5)
        x7 = self.conv4(x6)
        x8 = self.pool4(x7)
        # bottleneck
        x9 = self.conv5(x8)
        # Decoder
        x10 = self.tconv1(x9)
        concat_1  = torch.cat([crop_tensor(x7,x10), x10], dim=1)
        x11 = self.conv6(concat_1)
        x12 = self.tconv2(x11)
        concat_2  = torch.cat([crop_tensor(x5,x12), x12], dim=1)
        x13 = self.conv7(concat_2)
        x14 = self.tconv3(x13)
        concat_3  = torch.cat([crop_tensor(x3,x14), x14], dim=1)
        x15 = self.conv8(concat_3)
        x15 = self.tconv4(x15)
        concat_4  = torch.cat([crop_tensor(x1,x15), x15], dim=1)
        x16 = self.conv9(concat_4)
        out = self.out(x16)
        return out


if __name__ == "__main__":
    image = torch.rand(4, 3, 572, 572)
    model = Unet()
    out = model(image.shape)
