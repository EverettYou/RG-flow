import torch
from math import prod
from .nn import MarkedCNN, Dynamic
from .ode import ODEBijector

class RGPartition(torch.nn.Module):
    ''' RGPartition implements bijective partition of fine-grained features
        to coarse-grained and residual features.
        
        Example:
        par = RGPartition([5,5])
        x = torch.randn(1,1,5,5)
        y, z = par.split(x)
        x, y, z, par.merge(y, z)

        Parameters:
            in_shape :: torch.Size - data shape of fine-grained features
            stride :: int - stride of partition
    '''
    def __init__(self, in_shape, stride=2, **kwargs):
        assert stride>=2, f'stride must be greater equal than 2, but got {stride}.'
        super().__init__()
        self.in_shape = torch.Size(in_shape)
        out_shape = []
        self._mask = torch.tensor([True])
        for size in in_shape:
            axial_mask = torch.arange(size)%stride==stride//2
            self._mask = torch.kron(self._mask, axial_mask)
            out_shape.append(torch.count_nonzero(axial_mask))
        self.out_shape = torch.Size(out_shape)
        self.register_buffer('out_indx', torch.nonzero(self._mask).view(-1))
        self.register_buffer('res_indx', torch.nonzero(torch.logical_not(self._mask)).view(-1))
        self.res_shape = self.res_indx.shape
            
    def extra_repr(self):
        return f'{tuple(self.in_shape)} -> {tuple(self.out_shape)}, {tuple(self.res_shape)}'
        
    @property
    def mask(self):
        return self._mask.view(self.in_shape)
    
    def split(self, x):
        ''' split x to x, z
            Input:
                x :: torch.Tensor (N, dim, *in_shape) - fine-grained features
            Output:
                x :: torch.Tensor (N, dim, *out_shape) - coarse-grained features
                z :: torch.Tensor (N, dim, *res_shape) - residual (irrelevant) features
        '''
        assert prod(x.shape[2:]) == prod(self.in_shape), f'Data shape of x:{x.shape[2:]} does not match the designated shape {self.in_shape}.'
        x = x.view(x.shape[:2]+(-1,)) # (N, dim, :)
        x, z = x[:,:,self.out_indx], x[:,:,self.res_indx] 
        x = x.view(x.shape[:2]+self.out_shape)
        return x, z
    
    def merge(self, x, z):
        ''' merge x, z to x
            Input:
                x :: torch.Tensor (N, dim, *out_shape) - coarse-grained features
                z :: torch.Tensor (N, dim, *res_shape) - residual (irrelevant) features
            Output:
                x :: torch.Tensor (N, dim, *in_shape) - fine-grained features
        '''
        assert prod(x.shape[2:]) == prod(self.out_shape), f'Data shape of x:{x.shape[2:]} does not match the designated shape {self.out_shape}.'
        assert prod(z.shape[2:]) == prod(self.res_shape), f'Data shape of z:{z.shape[2:]} does not match the designated shape {self.res_shape}.'
        x = x.view(x.shape[:2]+(-1,)) # (N, dim, :)
        indx = torch.cat([self.out_indx, self.res_indx])
        x = torch.cat([x, z], -1)
        x[:,:,indx] = x.clone()
        x = x.view(x.shape[:2]+self.in_shape)
        return x

class RGLayer(torch.nn.Module):
    ''' perform one layer of RG transformation (bijective)
        
        Parameter:
            in_shape :: torch.Size - data shape of input
            dim :: int - feature dimensions
            --- optional ---
            hdims :: list of int - hidden feature dimensions (for all hidden layers)
            ksize :: int - kernel size (should be odd)
            stride :: int - stride of partition
            bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
            bias :: bool - If True, adds a learnable bias to the output
            activation :: str - activation function name
            hyper_dim :: int - latent space dimension of hyper net
    '''
    def __init__(self, in_shape, dim, **kwargs):
        super().__init__()
        self.partitioner = RGPartition(in_shape, **kwargs)
        self.in_shape = self.partitioner.in_shape
        self.out_shape = self.partitioner.out_shape
        self.res_shape = self.partitioner.res_shape
        self.dim = dim
        self.bijector = ODEBijector(Dynamic(MarkedCNN(self.partitioner.mask, dim, **kwargs), **kwargs))
            
    def encode(self, x, **kwargs):
        ''' single-layer encoding (renormalization) map x, z = R(x)
            Input:
                x :: torch.Tensor (N, dim, *in_shape) - fine-grained features
                --- optional ---
                mode :: str - computation mode 'f', 'jf' or 'jf_reg'
                div :: str - method for divergence estimation 'approx' or 'exact'
            Output:
                x :: torch.Tensor (N, dim, *out_shape) - coarse-grained features
                z :: torch.Tensor (N, dim, *res_shape) - residual (irrelevant) features
                --- optional ---
                logJ :: torch.Tensor (N,) - accumulated log Jacobian (log det d(x,z)/dx)
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        x, *rest = self.bijector(x, 0., 1., **kwargs) # bijector forward
        x, z = self.partitioner.split(x) # split data
        return x, z, *rest
    
    def decode(self, x, z, **kwargs):
        ''' single-layer decoding (generation) map x = G(x, z)
            Input:
                x :: torch.Tensor (N, dim, *out_shape) - coarse-grained features
                z :: torch.Tensor (N, dim, *res_shape) - residual (irrelevant) features
                --- optional ---
                mode :: str - computation mode 'f', 'jf' or 'jf_reg'
                div :: str - method for divergence estimation 'approx' or 'exact'
            Output:
                x :: torch.Tensor (N, dim, *in_shape) - fine-grained features
                --- optional ---
                logJ :: torch.Tensor (N,) - accumulated log Jacobian (log det dx/d(x,z))
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        x = self.partitioner.merge(x, z) # merge data
        x, *rest = self.bijector(x, 1., 0., **kwargs) # bijector backward
        return x, *rest

class RGFlow(torch.nn.Module):
    ''' perform the RG flow (bijective)
        
        Parameter:
            shape :: torch.Size - data shape
            dim :: int - feature dimensions
            --- optional ---
            hdims :: list of int - hidden feature dimensions (for all hidden layers)
            ksize :: int - kernel size (should be odd)
            stride :: int - stride of partition
            bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
            bias :: bool - If True, adds a learnable bias to the output
            activation :: str - activation function name
            hyper_dim :: int - latent space dimension of hyper net
    '''
    def __init__(self, shape, dim, **kwargs):
        super().__init__()
        self.in_shape = torch.Size(shape)
        self.out_shape = torch.Size([prod(shape)])
        self.dim = dim
        self.layers = torch.nn.ModuleList()
        while prod(shape) > 0:
            layer = RGLayer(shape, dim, **kwargs)
            self.layers.append(layer)
            shape = layer.out_shape
    
    def encode(self, x, **kwargs):
        ''' encoding (renormalization) map z = R(x)
            Input:
                x :: torch.Tensor (N, dim, *shape) - boundary features
                --- optional ---
                mode :: str - computation mode 'f', 'jf' or 'jf_reg'
                div :: str - method for divergence estimation 'approx' or 'exact'
            Output:
                z :: torch.Tensor (N, dim, *shape) - bulk features
                --- optional ---
                logJ :: torch.Tensor (N,) - accumulated accumulated log Jacobian (log det dz/dx)
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        rest_acc = None
        zs = []
        for layer in self.layers:
            x, z, *rest = layer.encode(x, **kwargs)
            zs.append(z)
            if rest_acc is None:
                rest_acc = rest
            else:
                rest_acc = tuple(R+r for R, r in zip(rest_acc, rest))
        z = torch.cat(zs,-1)
        return z, *rest_acc
    
    def decode(self, z, **kwargs):
        ''' decoding (generation) map x = G(z)
            Input:
                z :: torch.Tensor (N, dim, *shape) - bulk features
                --- optional ---
                mode :: str - computation mode 'f', 'jf' or 'jf_reg'
                div :: str - method for divergence estimation 'approx' or 'exact'
            Output:
                x :: torch.Tensor (N, dim, *shape) - boundary features
                --- optional ---
                logJ :: torch.Tensor (N,) - accumulated accumulated log Jacobian (log det dx/dz)
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        rest_acc = None
        x = z[:, :, []]
        pz = z.shape[-1]
        for layer in reversed(self.layers):
            nz = prod(layer.res_shape)
            x, *rest = layer.decode(x, z[:, :, pz-nz:pz], **kwargs)
            pz -= nz
            if rest_acc is None:
                rest_acc = rest
            else:
                rest_acc = tuple(R+r for R, r in zip(rest_acc, rest))
        return x, *rest_acc

