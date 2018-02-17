import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import numpy as np

from revnet import (rev_block,
                    rev_block_function)
from fcn_maker.blocks import convolution


class simple_block(rev_block):
    def __init__(self, in_channels, out_channels, activations,
                 subsample=False):
        f_modules = [convolution(in_channels=in_channels//2,
                                 out_channels=out_channels//2,
                                 kernel_size=3,
                                 ndim=2,
                                 init='kaiming_normal',
                                 padding=1)]
        g_modules = [convolution(in_channels=in_channels//2,
                                 out_channels=out_channels//2,
                                 kernel_size=3,
                                 ndim=2,
                                 init='kaiming_normal',
                                 padding=1)]
        super(simple_block, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           activations=activations,
                                           f_modules=f_modules,
                                           g_modules=g_modules,
                                           subsample=False)
        

class simple_net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(simple_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activations = []
        self.layers = [simple_block(in_channels=in_channels,
                                    out_channels=out_channels,
                                    activations=self.activations)]
        
    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(x)
            
        # Save last output for backward
        self.activations.append(out.data)
        
        return out


class trivial_module(torch.nn.Module):
    def forward(self, x):
        return x*2

if __name__=='__main__':
    print("TESTING FORWARD PASS")
    model = simple_net(in_channels=4, out_channels=4)
    data = np.random.rand(5,4,25,25).astype(np.float32)
    data = Variable(torch.from_numpy(data))
    try:
        out = model(data)
    except:
        print("> FAIL")
        raise
    if data.size()==out.size():
        print("> OK : Successfully returned output of correct size.")
    else:
        print("> FAIL : For input of size {}, returned output of "
              "wrong size {}.".format(data.size(), out.size()))
    
    print("TESTING INPUT RECREATION (DIRECT FORWARD & BACKWARD)")
    data = np.random.rand(5,4,25,25).astype(np.float32)
    #data = np.ones((1,4,1,1), dtype=np.float32)
    data = torch.from_numpy(data)
    f_modules = [convolution(in_channels=2,
                             out_channels=2,
                             kernel_size=3,
                             ndim=2,
                             init='kaiming_normal',
                             padding=2,
                             dilation=2)]
    g_modules = [convolution(in_channels=2,
                             out_channels=2,
                             kernel_size=3,
                             ndim=2,
                             init='kaiming_normal',
                             padding=2,
                             dilation=2)]
    #f_modules = [trivial_module()]
    #g_modules = [trivial_module()]
    try:
        y = rev_block_function._forward(data,
                                        in_channels=2,
                                        out_channels=2,
                                        f_modules=f_modules,
                                        g_modules=g_modules)
        z = rev_block_function._backward(y.data,
                                         in_channels=2,
                                         out_channels=2,
                                         f_modules=f_modules,
                                         g_modules=g_modules)
    except:
        print("> FAIL")
        raise
    equal = torch.lt(torch.abs(data-z), 1e-5).all()
    if equal:
        print("> OK : Input recreated successfully!")
    else:
        print("> FAIL : Output differs from input.")

    print("TESTING AUTOGRAD BACKWARD")
    model = simple_net(in_channels=4, out_channels=4)
    data = np.random.rand(5,4,25,25).astype(np.float32)
    data = Variable(torch.from_numpy(data))
    out = model(data)
    out = out.mean()
    try:
        out.backward()
    except:
        print("> FAIL")
        raise
    print("> OK : Backward computed without raising an exception.")
