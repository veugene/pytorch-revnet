import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable

from fcn_maker.blocks import convolution


def unpack_modules(module_stack):
    params = []
    buffs = []
    for m in module_stack:
        params.extend(m.parameters())
        buffs.extend(m._all_buffers())
    return tuple(params), tuple(buffs)


def possible_downsample(x, in_channels, out_channels, subsample=False):
    out = x

    # Downsample image
    if subsample:
        out = F.avg_pool2d(out, 2, 2)

    # Pad with empty channels
    if in_channels < out_channels:
        pad = Variable(torch.zeros(out.size(0),
                                   (out_channels - in_channels) // 2,
                                   out.size(2), out.size(3)),
                       requires_grad=True)
        temp = torch.cat([pad, out], dim=1)
        out = torch.cat([temp, pad], dim=1)

    ## If we did nothing, add zero tensor, so the output of this function
    ## depends on the input in the graph
    #try: out
    #except:
        #injection = Variable(torch.zeros_like(x.data), requires_grad=True)

        #out = x + injection

    return out


class rev_block_function(Function):
    @staticmethod
    def residual(x, in_channels, out_channels, modules):
        """Compute a pre-activation residual function.

        Args:
            x (Variable): The input variable
            in_channels (int): Number of channels of x
            out_channels (int): Number of channels of the output

        Returns:
            out (Variable): The result of the computation

        """
        out = x
        for m in modules:
            out = m(out)
        return out

    @staticmethod
    def _forward(x, in_channels, out_channels, f_modules, g_modules,
                 subsample=False):

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.no_grad():
            x1 = Variable(x1.contiguous())
            x2 = Variable(x2.contiguous())

            x1_ = possible_downsample(x1, in_channels, out_channels, subsample)
            x2_ = possible_downsample(x2, in_channels, out_channels, subsample)

            f_x2 = rev_block_function.residual(x2,
                                               in_channels,
                                               out_channels,
                                               f_modules)
            
            y1 = f_x2 + x1_
            
            g_y1 = rev_block_function.residual(y1,
                                               out_channels,  # FIX ??
                                               out_channels,
                                               g_modules)

            y2 = g_y1 + x2_
            
            y = torch.cat([y1, y2], dim=1)

            del y1, y2
            del x1, x2

        return y

    @staticmethod
    def _backward(output, in_channels, out_channels, f_modules, g_modules):

        y1, y2 = torch.chunk(output, 2, dim=1)
        with torch.no_grad():
            y1 = Variable(y1.contiguous())
            y2 = Variable(y2.contiguous())

            x2 = y2 - rev_block_function.residual(y1,
                                                  out_channels, # FIX ??
                                                  out_channels,
                                                  g_modules)

            x1 = y1 - rev_block_function.residual(x2,
                                                  in_channels,
                                                  out_channels,
                                                  f_modules)

            del y1, y2
            x1, x2 = x1.data, x2.data

            x = torch.cat((x1, x2), 1)
        return x

    @staticmethod
    def _grad(x, dy, in_channels, out_channels, f_modules, g_modules,
              activations, subsample=False):
        dy1, dy2 = Variable.chunk(dy, 2, dim=1)

        x1, x2 = torch.chunk(x, 2, dim=1)

        with torch.enable_grad():
            x1 = Variable(x1.contiguous(), requires_grad=True)
            x2 = Variable(x2.contiguous(), requires_grad=True)
            x1.retain_grad()
            x2.retain_grad()

            x1_ = possible_downsample(x1, in_channels, out_channels, subsample)
            x2_ = possible_downsample(x2, in_channels, out_channels, subsample)

            f_x2 = rev_block_function.residual(x2,
                                               in_channels,
                                               out_channels,
                                               f_modules)

            y1_ = f_x2 + x1_

            g_y1 = rev_block_function.residual(y1_,
                                               out_channels,    # FIX ??
                                               out_channels,
                                               g_modules)

            y2_ = g_y1 + x2_
            
            f_params, f_buffs = unpack_modules(f_modules)
            g_params, g_buffs = unpack_modules(g_modules)

            dd1 = torch.autograd.grad(y2_, (y1_,) + tuple(g_params), dy2,
                                      retain_graph=True)
            dy2_y1 = dd1[0]
            dgw = dd1[1:]
            dy1_plus = dy2_y1 + dy1
            dd2 = torch.autograd.grad(y1_, (x1, x2) + tuple(f_params), dy1_plus,
                                      retain_graph=True)
            dfw = dd2[2:]

            dx2 = dd2[1]
            dx2 += torch.autograd.grad(x2_, x2, dy2, retain_graph=True)[0]
            dx1 = dd2[0]

            activations.append(x)

            y1_.detach_()
            y2_.detach_()
            del y1_, y2_
            dx = torch.cat((dx1, dx2), 1)

        return dx, dfw, dgw

    @staticmethod
    def forward(ctx, x, in_channels, out_channels, f_modules, g_modules,
                activations, subsample=False, *args):
        """Compute forward pass including boilerplate code.

        This should not be called directly, use the apply method of this class.

        Args:
            ctx (Context):                  Context object, see PyTorch docs
            x (Tensor):                     4D input tensor
            in_channels (int):              Number of channels on input
            out_channels (int):             Number of channels on output
            f_modules (List):               Sequence of modules for F function
            g_modules (List):               Sequence of modules for G function
            activations (List):             Activation stack
            subsample (bool):               Whether to do 2x spatial pooling
            *args:                          Should contain all the
                                            Parameters of the module
        """
        
        # if subsampling, information is lost and we need to save the input
        if subsample:
            activations.append(x)
            ctx.load_input = True
        else:
            ctx.load_input = False

        ctx.activations = activations
        ctx.in_channels = in_channels
        ctx.out_channels = out_channels
        ctx.f_modules = f_modules
        ctx.g_modules = g_modules
        ctx.subsample = subsample

        y = rev_block_function._forward(x,
                                        in_channels,
                                        out_channels,
                                        f_modules,
                                        g_modules,
                                        subsample)

        return y.data

    @staticmethod
    def backward(ctx, grad_out):
        # Load or reconstruct input
        if ctx.load_input:
            ctx.activations.pop()
            x = ctx.activations.pop()
        else:
            output = ctx.activations.pop()
            x = rev_block_function._backward(output,
                                             ctx.in_channels,
                                             ctx.out_channels,
                                             ctx.f_modules,
                                             ctx.g_modules)

        dx, dfw, dgw = rev_block_function._grad(x,
                                                grad_out,
                                                ctx.in_channels,
                                                ctx.out_channels,
                                                ctx.f_modules,
                                                ctx.g_modules,
                                                ctx.activations,
                                                ctx.subsample)

        return (dx,) + (None,)*6 + tuple(dfw) + tuple(dgw)


class rev_block(nn.Module):
    def __init__(self, in_channels, out_channels, activations,
                 f_modules, g_modules, subsample=False):
        super(rev_block, self).__init__()
        # NOTE: channels are only counted for possible_downsample()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.activations = activations
        self.f_modules = f_modules
        self.g_modules = g_modules
        self.subsample = subsample
        #self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x):
        # Unpack parameters and buffers
        f_params, f_buffs = unpack_modules(self.f_modules)
        g_params, g_buffs = unpack_modules(self.g_modules)
        
        return rev_block_function.apply(x,
                                        self.in_channels,
                                        self.out_channels,
                                        self.f_modules,
                                        self.g_modules,
                                        self.activations,
                                        self.subsample,
                                        *f_params,
                                        *g_params)


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
