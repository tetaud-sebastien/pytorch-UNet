from torch.nn import functional as F
import torch
from torch import nn

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, dilation=1, activation='relu'):
        """
        
        """
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation,
                              bias=self.use_bias)
    def forward(self, xin):
       
        x = self.conv(xin)
        x = self.activation(x)
        return x
        
class TransConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, dilation=1, activation='relu'):
        """
        
        """
        super(TransConv2dBlock, self).__init__()
        self.use_bias = True
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=dilation,
                              bias=self.use_bias)
    def forward(self, xin):
       
        x = self.conv(xin)
        x = self.activation(x)
        return x
    

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder  
        self.conv0 = Conv2dBlock(in_channels=3, out_channels=64, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        self.conv1 = Conv2dBlock(in_channels=64, out_channels=64, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2dBlock(in_channels=64, out_channels=128, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        self.conv3 = Conv2dBlock(in_channels=128, out_channels=128, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv4 = Conv2dBlock(in_channels=128, out_channels=256, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv5 = Conv2dBlock(in_channels=256, out_channels=256, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    
        self.conv6 = Conv2dBlock(in_channels=256, out_channels=512, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv7 = Conv2dBlock(in_channels=512, out_channels=512, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv8 = Conv2dBlock(in_channels=512, out_channels=1024, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv9 = Conv2dBlock(in_channels=1024, out_channels=1024, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        # Decoder
        
        self.upconv1 = TransConv2dBlock(in_channels=1024, out_channels=512, 
                                 kernel_size=2, stride=2,
                                 padding=0, dilation=1, activation='relu')


        self.conv10 = Conv2dBlock(in_channels=1024, out_channels=512, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv11 = Conv2dBlock(in_channels=512, out_channels=512, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        

        self.upconv2 = TransConv2dBlock(in_channels=512, out_channels=256, 
                                 kernel_size=2, stride=2,
                                 padding=0, dilation=1, activation='relu')

        self.conv12 = Conv2dBlock(in_channels=512, out_channels=256, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv13 = Conv2dBlock(in_channels=256, out_channels=256, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        

        self.upconv3 = TransConv2dBlock(in_channels=256, out_channels=128, 
                                 kernel_size=2, stride=2,
                                 padding=0, dilation=1, activation='relu')

        self.conv14 = Conv2dBlock(in_channels=256, out_channels=128, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv15 = Conv2dBlock(in_channels=128, out_channels=128, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.upconv4 = TransConv2dBlock(in_channels=128, out_channels=64, 
                                 kernel_size=2, stride=2,
                                 padding=0, dilation=1, activation='relu')

        self.conv16 = Conv2dBlock(in_channels=128, out_channels=64,
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv17 = Conv2dBlock(in_channels=64, out_channels=64, 
                                 kernel_size=3, stride=1,
                                 padding=0, dilation=1, activation='relu')
        
        self.conv18 = Conv2dBlock(in_channels=64, out_channels=2, 
                                 kernel_size=1, stride=1,
                                 padding=0, dilation=1, activation='relu')

     
        
    def forward(self, x):

        # Encoder
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        p1 = self.pool1(conv1)
        conv2 = self.conv2(p1)
        conv3 = self.conv3(conv2)
        p2 = self.pool2(conv3)
        conv4 = self.conv4(p2)
        conv5 = self.conv5(conv4)
        p3 = self.pool3(conv5)
        conv6 = self.conv6(p3)
        conv7 = self.conv7(conv6)
        p4 = self.pool4(conv7)
        conv8 = self.conv8(p4)
        conv9 = self.conv9(conv8)
      
        # Decoder
        upconv1 = self.upconv1(conv9)
        conv7_resize = F.interpolate(conv7, size=upconv1.shape[2], mode='nearest')
        concat_1  = torch.cat([conv7_resize, upconv1], dim=1)
        conv10 = self.conv10(concat_1)
        conv11 = self.conv11(conv10)
        upconv2 = self.upconv2(conv11)
        conv5_resize = F.interpolate(conv5, size=upconv2.shape[2], mode='nearest')
        concat_2  = torch.cat([conv5_resize, upconv2], dim=1)
        conv12 = self.conv12(concat_2)
        conv13 = self.conv13(conv12)
        upconv3 = self.upconv3(conv13)
        conv3_resize = F.interpolate(conv3, size=upconv3.shape[2], mode='nearest')
        concat_3  = torch.cat([conv3_resize, upconv3], dim=1)
        conv14 = self.conv14(concat_3)
        conv15 = self.conv15(conv14)
        upconv4 = self.upconv4(conv15)
        conv4_resize = F.interpolate(conv1, size=upconv4.shape[2], mode='nearest')
        concat_4  = torch.cat([conv4_resize, upconv4], dim=1)
        conv16 = self.conv16(concat_4)
        conv17 = self.conv17(conv16)
        conv18 = self.conv18(conv17)
        return conv18

if __name__ == '__main__':


    image = torch.rand(4, 3, 572, 572)
    model = Unet()
    out = model(image)
    print(out.shape)

    # out = out[0,:,:,:]
    # class_probs = torch.softmax(out, dim=0)
    # mask_prediction = torch.argmax(class_probs, dim=0).cpu().numpy()