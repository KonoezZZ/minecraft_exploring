from . import prerequisites
from . import data_processing



##### load your data #####
train_set = torch.load(r'YOUR_PATH/LOADER_NAME_train.pt')  
valid_set = torch.load(r'YOUR_PATH/LOADER_NAME_valid')  



##### baseline #####
available_actions = [[0, 0, 0],   # no action
            [0, 0, 1],   # jump
            [0, 1, 0],   # right
            [0, 1, 1],   # right+jump
            [0, -1, 0],  # left
            [0, -1, 1],  # left+jump
            [1, 0, 0],   # forward
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, -1, 0],
            [1, -1, 1],
            [-1, 0, 0],  # backward
            [-1, 0, 1],
            [-1, 1, 0],
            [-1, 1, 1],
            [-1, -1, 0],
            [-1, -1, 1]]

act_train = train_set[1].squeeze().numpy()
act_train_classes = np.full((len(act_train)), -1, dtype=np.int)
act_val = valid_set[1].squeeze().numpy()
act_val_classes = np.full((len(act_val)), -1, dtype=np.int)

for i, a in enumerate(available_actions):
    act_train_classes[np.all(act_train == a, axis=1)] = i
    act_val_classes[np.all(act_val == a, axis=1)] = i
    
act_train = torch.tensor(act_train_classes).int()
act_val = torch.tensor(act_val_classes).int()
(values, counts) = np.unique(act_train, return_counts=True)
ind = np.argmax(counts)

print('Accuracy: {:.4f}'.format(np.count_nonzero(act_val==values[ind])/len(act_val)))


##### Simple CNN #####
"""
Implementation based on:
https://github.com/ivan-vasilev/Python-Deep-Learning-SE/ch10/imitation_learning/train.py
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def build_network_s2a(num_available_actions=18):
    """Build the torch network"""
    # Same network as with the DQN example
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 8, 4),
        torch.nn.BatchNorm2d(32),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Conv2d(32, 64, 4, 2),
        torch.nn.BatchNorm2d(64),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.5),
        torch.nn.Conv2d(64, 64, 3, 1),
        torch.nn.ELU(),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(1024),
        torch.nn.Dropout(),
        torch.nn.Linear(1024, 120),
        torch.nn.ELU(),
        torch.nn.BatchNorm1d(120),
        torch.nn.Dropout(),
        torch.nn.Linear(120, num_available_actions),
        #torch.nn.Softmax(),
    )

    return model


def train(model, device, train_loader, valid_loader, EPOCHS=30, path='default.pt'):
    """
    Training main method
    :param model: the network
    :param device: the cuda device
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # train
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, train_acc = train_epoch(model,
                          device,
                          loss_function,
                          optimizer,
                          train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_loss, test_acc = test(model, 
                      device,
                      loss_function,
                      valid_loader)
        valid_losses.append(test_loss)
        valid_accs.append(test_acc)
        if valid_losses[-1] == np.min(valid_losses):
            # save model
            torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path))
    return model, [train_losses, valid_losses, train_accs, valid_accs]

        
def train_epoch(model, device, loss_function, optimizer, data_loader):
    """Train for a single epoch"""
    # set model to training mode
    model.train()
    current_loss = 0.0
    current_acc = 0
    # iterate over the training data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        inputs = rgb2gray(inputs).unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs.float(), labels.long())  # notice the data types
            # backward
            loss.backward()
            optimizer.step()
        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    print('Train Loss: {:.4f}; Accuracy: {:.4f}'.format(total_loss, total_acc))
    return total_loss, total_acc

    
def test(model, device, loss_function, data_loader):
    """Test over the whole dataset"""
    model.eval()  # set model in evaluation mode
    current_loss = 0.0
    current_acc = 0
    # iterate over the validation data
    for i, (inputs, labels) in enumerate(data_loader):
        # send the input/labels to the GPU
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        inputs = rgb2gray(inputs).unsqueeze(1)
        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            loss = loss_function(outputs.float(), labels.long())
        # statistics
        current_loss += loss.item() * inputs.size(0)
        current_acc += torch.sum(predictions == labels.data)
    total_loss = current_loss / len(data_loader.dataset)
    total_acc = current_acc.double() / len(data_loader.dataset)
    print('Test Loss: {:.4f}; Accuracy: {:.4f}'
          .format(total_loss, total_acc))
    return total_loss, total_acc
  
  def train_simple():
    MODE = 's2a'
    BATCH_SIZE = 32
    #train_set, valid_set = dataset.train_valid_split(num_samples=1000)
    train_set0, valid_set0 = train_set.copy(), valid_set.copy()
    
    act_train = train_set0[1].squeeze().numpy()
    act_train_classes = np.full((len(act_train)), -1, dtype=np.int)
    for i, a in enumerate(available_actions):
        act_train_classes[np.all(act_train == a, axis=1)] = i
    train_set0[1] = torch.tensor(act_train_classes).int()

    act_val = valid_set0[1].squeeze().numpy()
    act_val_classes = np.full((len(act_val)), -1, dtype=np.int)
    for i, a in enumerate(available_actions):
        act_val_classes[np.all(act_val == a, axis=1)] = i
    valid_set0[1] = torch.tensor(act_val_classes).int()
    
    train_loader, valid_loader = get_loaders(train_set0, valid_set0, MODE, BATCH_SIZE)
    
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_network_s2a()
    model.eval()
    model.to(dev)
    model, metrics = train(model, dev, train_loader, valid_loader, EPOCHS=20, path='s2a_default.pt') 
    return model, metrics

  
trained_simple, metrics = train_simple()



##### ResNet #####
"""
Implementation based on 
https://github.com/FrancescoSaverioZuppichini/ResNet
"""
import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False) 


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

  
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None
           
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))

  
class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
            
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
      

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)
  
def resnet(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):   # this is the mini-resnet
    return ResNet(in_channels, n_classes, block=block, deepths=[1, 1], *args, **kwargs)


def train_resnet():
    MODE = 's2a'
    BATCH_SIZE = 64
    #train_set, valid_set = dataset.train_valid_split(num_samples=1000)
    train_set0, valid_set0 = train_set.copy(), valid_set.copy()
    
    available_actions = [[0, 0, 0],   # no action
                [0, 0, 1],   # jump
                [0, 1, 0],   # right
                [0, 1, 1],   # right+jump
                [0, -1, 0],  # left
                [0, -1, 1],  # left+jump
                [1, 0, 0],   # forward
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, -1, 0],
                [1, -1, 1],
                [-1, 0, 0],  # backward
                [-1, 0, 1],
                [-1, 1, 0],
                [-1, 1, 1],
                [-1, -1, 0],
                [-1, -1, 1]]

    act_train = train_set0[1].squeeze().numpy()
    act_train_classes = np.full((len(act_train)), -1, dtype=np.int)
    for i, a in enumerate(available_actions):
        act_train_classes[np.all(act_train == a, axis=1)] = i
    train_set0[1] = torch.tensor(act_train_classes).int()

    act_val = valid_set0[1].squeeze().numpy()
    act_val_classes = np.full((len(act_val)), -1, dtype=np.int)
    for i, a in enumerate(available_actions):
        act_val_classes[np.all(act_val == a, axis=1)] = i
    valid_set0[1] = torch.tensor(act_val_classes).int()
    
    train_loader, valid_loader = get_loaders(train_set0, valid_set0, MODE, BATCH_SIZE)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(3, len(available_actions))
    model.eval()
    model.to(dev)
    model, metrics = train(model, dev, train_loader, valid_loader, EPOCHS=10, path='MODEL_NAME.pt')
    return model, metrics

trained_resnet, metrics = main()
