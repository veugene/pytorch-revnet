import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .blocks import rev_block


class basic_block(rev_block):
    def __init__(self, in_channels, out_channels, activations,
                 subsample=False):
        super(basic_block, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          activations=activations,
                                          subsample=subsample)
        #self.add_module(nn.BatchNorm2d, self.in_channels)
        self.add_module(nn.ReLU)
        self.add_module(nn.Conv2d,
                        in_channels=in_channels//2,
                        out_channels=out_channels//2,
                        kernel_size=3,
                        padding=1,
                        stride=2 if subsample else 1)
        #self.add_module(nn.BatchNorm2d, self.out_channels)
        self.add_module(nn.ReLU)
        self.add_module(nn.Conv2d,
                        in_channels=out_channels//2,
                        out_channels=out_channels//2,
                        kernel_size=3,
                        padding=1)


class revnet(nn.Module):
    def __init__(self, units, filters, subsample, classes):
        """
        Parameters
        ----------

        units: list-like
            Number of residual units in each group

        filters: list-like
            Number of filters in each unit including the inputlayer, so it is
            one item longer than units

        subsample: list-like
            List of boolean values for each block, specifying whether it should do 2x spatial subsampling

        bottleneck: boolean
            Wether to use the bottleneck residual or the basic residual
        """
        super(revnet, self).__init__()
        self.name = self.__class__.__name__
        self.activations = []
        self.block = basic_block
        self.layers = nn.ModuleList()

        # Input layers
        self.layers.append(nn.Conv2d(3, filters[0], 3, padding=1))
        self.layers.append(nn.BatchNorm2d(filters[0]))
        self.layers.append(nn.ReLU())

        for i, group in enumerate(units):
            self.layers.append(self.block(filters[i], filters[i + 1],
                                          self.activations,
                                          subsample=subsample[i]))
            for unit in range(1, group):
                self.layers.append(self.block(filters[i + 1],
                                              filters[i + 1],
                                              self.activations))
        self.bn_last = nn.BatchNorm2d(filters[-1])
        self.fc = nn.Linear(filters[-1], classes)
        
    def cuda(self, device=None):
        for m in self.layers:
            m.cuda()
        return self._apply(lambda t: t.cuda(device))
    
    def cpu(self):
        for m in self.layers:
            m.cpu()
        return self._apply(lambda t: t.cpu())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        self.activations.append(x.data)
        x = F.relu(self.bn_last(x))
        x = F.avg_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def free(self):
        del self.activations[:]
        
        
def revnet38():
    model = revnet(units=[3, 3, 3],
                   filters=[32, 32, 64, 112],
                   subsample=[0, 1, 1],
                   classes=10)
    model.name = "revnet38"
    return model


def revnet110():
    model = revnet(units=[9, 9, 9],
                   filters=[32, 32, 64, 112],
                   subsample=[0, 1, 1],
                   classes=10)
    model.name = "revnet110"
    return model
