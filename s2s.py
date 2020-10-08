from . import prerequisites
from . import data_processing



##### load your data #####
train_set = torch.load(r'YOUR_PATH/LOADER_NAME_train.pt')  
valid_set = torch.load(r'YOUR_PATH/LOADER_NAME_valid')  



##### baseline #####
MODE = 's2s'
BATCH_SIZE = 32
train_loader, valid_loader = get_loaders(train_set, valid_set, MODE, BATCH_SIZE)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

losses = []
for epoch in range(1):
    for i, data in enumerate(valid_loader):
        X, y = data[0], data[1]
        X, y = Variable(X), Variable(y)
        losses.append(F.mse_loss(X * 255, y * 255).item())
        if i == 10000:
            break

print("valid loss:", np.mean(losses))



##### U-Net #####
"""
Implementation based on
https://github.com/jaxony/unet-pytorch/blob/master/model.py
"""

MODE = 's2s'
BATCH_SIZE = 32

train_loader, valid_loader = get_loaders(train_set, valid_set, MODE, BATCH_SIZE)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from skimage.morphology import binary_opening, disk, label
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

%matplotlib inline


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     groups=groups,
                     stride=1)


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,
                                  out_channels,
                                  kernel_size=2,
                                  stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))
        

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=False):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

      
class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels,
                                 self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet_s2s(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='upsample',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet_s2s, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []
        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)
        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)
        self.conv_final = conv1x1(outs, self.num_classes)
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)
    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)
    def forward(self, x):
        encoder_outs = []
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return F.sigmoid(x)


def get_loss(dl, model):
    loss = 0
    for data in dl:
        X, y = data[0], data[1]
        X, y = Variable(X), Variable(y)
        output = model(X)
        #loss += F.binary_cross_entropy(output, y)
        loss += F.mse_loss(output * 255, y * 255)
    loss = loss / len(dl)
    return loss


def plot_reconstructions(model, images, batch_idx):
    reconstructed = model(images)
    images = images.cpu().data[0]
    reconstructed = reconstructed.cpu().data[0]
    images = np.array(images.permute(1, 2, 0))
    reconstructed = np.array(reconstructed.permute(1, 2, 0))
    plt.imshow(images)
    plt.savefig(str(batch_idx) + "original_image.png")
    plt.close()
    plt.imshow(reconstructed)
    plt.savefig(str(batch_idx) + "reconstructed_images.png")
    plt.close()
    

## train a U-Net ##

class param:
    img_size = (64, 64)
    bs = 8
    num_workers = 4
    lr = 0.001
    epochs = 3
    unet_depth = 2
    unet_start_filters = 8
    log_interval = 100 # less then len(train_dl)


model = UNet_s2s(3,
        depth=param.unet_depth,
        start_filts=param.unet_start_filters,
        merge_mode='concat')
optim = torch.optim.Adam(model.parameters(), lr=param.lr)

iters = []
train_losses = []
val_losses = []

it = 0
min_loss = 999

train_losses = []
valid_losses = []

model.train()
for epoch in range(2):
    for i, data in enumerate(train_loader):
        X, y = data[0], data[1]
        X = Variable(X)  # [N, 1, H, W]
        y = Variable(y)  # [N, H, W] with class indices (0, 1)
        output = model(X)  # [N, 3, H, W]
        loss = F.mse_loss(output * 255, y * 255)
        #loss = F.binary_cross_entropy(output, y)
        train_losses.append(loss)

        if i % 50 == 0:
          print('epoch: {}, batch: {}, train loss: {}'.format(epoch, i, loss))
          plot_reconstructions(model=model, images=X, batch_idx=i)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i + 1) % param.log_interval == 0:
        #if (i + 1) % 1 == 0:
            #it += param.log_interval * param.bs
            #iters.append(it)
            #train_losses.append(loss.item())

            #model.eval()
            losses = []
            for i, data in enumerate(valid_loader):
                X, y = data[0], data[1]
                X, y = Variable(X), Variable(y)
                output = model(X)
                valid_loss = F.mse_loss(output * 255, y * 255).item()
                losses.append(valid_loss)
                if i == 10000:
                  break

            val_loss = np.mean(losses)
            valid_losses.append(val_loss)
            print('valid_loss: {}'.format(val_loss))

            if val_loss < min_loss:
                torch.save(model.state_dict(), 'YOUR_PATH/MODEL_NAME.pt')
                min_loss = val_loss
