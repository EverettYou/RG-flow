import torch
from math import prod
from .nn import CNN, Dynamic
from .ode import ODEBijector
from .hmc import LatentHMCSampler

class RGLayer(torch.nn.Module):
    ''' RGLayer implements bijective partition of fine-grained features
        to coarse-grained and residual features.
        
        Example:
        layer = RGLayer(0, [5,5])
        x = torch.randn(1,1,5,5)
        y, z = layer.split(x)
        x, y, z, layer.merge(y, z)

        Parameters:
            layer_id :: int - layer id (used to infer pixel type encoding)
            in_shape :: torch.Size - data shape of fine-grained features
            stride :: int - stride of partition
    '''
    def __init__(self, layer_id, in_shape, stride=2, **kwargs):
        assert stride>=2, f'stride must be greater equal than 2, but got {stride}.'
        super().__init__()
        self.layer_id = layer_id
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
    def type_code(self):
        ''' type encoding of pixels in this layer '''
        code = self._mask.to(torch.long) + self.layer_id * 2
        return code.view(self.in_shape)
    
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

class RGFlow(torch.nn.Module):
    ''' perform the RG flow (bijective)
        
        Parameter:
            shape :: torch.Size - data shape
            dim :: int - feature dimensions
            type_dim :: int - type encoding dimension
            --- optional ---
            hdims :: list of int - hidden feature dimensions (for all hidden layers)
            ksize :: int - kernel size (should be odd)
            stride :: int - stride of partition
            bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
            bias :: bool - If True, adds a learnable bias to the output
            activation :: str - activation function name
            hyper_dim :: int - latent space dimension of hyper net
    '''
    def __init__(self, shape, dim, type_dim, hdims=[], **kwargs):
        super().__init__()
        self.in_shape = torch.Size(shape)
        self.out_shape = torch.Size([prod(shape)])
        self.dim = dim
        self.layers = []
        layer_id = 0
        while prod(shape) > 0:
            layer = RGLayer(layer_id, shape, **kwargs)
            self.layers.append(layer)
            shape = layer.out_shape
            layer_id += 1
        self.type_embd = torch.nn.Embedding(layer_id * 2, type_dim)
        d = len(self.in_shape)
        dims = [dim + type_dim] + hdims + [dim]
        self.bijector = ODEBijector(Dynamic(CNN(d, dims, **kwargs), **kwargs))

    def get_bg(self, layer, xshape):
        ''' helper function to obtain background tensor at each layer given shape of input '''
        bg = self.type_embd(layer.type_code) # (*shape, type_dim)
        bg = bg.permute(torch.arange(bg.ndim).roll(1).tolist()) # (type_dim, *shape)
        bg = bg.unsqueeze(0).expand((xshape[0],-1)+xshape[2:]) # (N, type_dim, *shape)
        return bg

    def encode(self, x, **kwargs):
        ''' encoding (renormalization) map z = R(x)
            Input:
                x :: torch.Tensor (N, dim, *shape) - boundary features
                --- optional ---
                div :: str - method for divergence estimation
                    (this is only used if jac or reg is True)
                    'approx' : approximate method
                    'exact' : exact method
                jac :: bool - whether to calculate differential Jacobian 
                reg :: bool - whether to calculate differential energies
                --- ODE solver parameters ---
                rtol: upper bound on relative error
                atol: upper bound on absolute error
                method: integration method, default 'dopri5'
                (see https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py)

            Output:
                z   :: torch.Tensor (same shape as input) - the output
                jac :: torch.Tensor (N,) - log Jacobian of output with respect to input
                reg :: torch.Tensor (M, 2) - regularization energies (Ek, Eg)
        '''
        rest_acc = None
        zs = []
        for layer in self.layers:
            bg = self.get_bg(layer, x.shape) # (N, type_dim, *shape)
            x, *rest = self.bijector.forward(x, bg=bg, **kwargs)
            x, z = layer.split(x)
            zs.append(z)
            if rest_acc is None:
                rest_acc = rest
            else:
                rest_acc = tuple(a + b for a, b in zip(rest_acc, rest))
        z = torch.cat(zs,-1)
        if len(rest_acc) == 0:
            return z
        else:
            return z, *rest_acc
    
    def decode(self, z, **kwargs):
        ''' decoding (generation) map x = G(z)
            Input:
                z :: torch.Tensor (N, dim, *shape) - bulk features
                --- optional ---
                div :: str - method for divergence estimation
                    (this is only used if jac or reg is True)
                    'approx' : approximate method
                    'exact' : exact method
                jac :: bool - whether to calculate differential Jacobian 
                reg :: bool - whether to calculate differential energies
                --- ODE solver parameters ---
                rtol: upper bound on relative error
                atol: upper bound on absolute error
                method: integration method, default 'dopri5'
                (see https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py)

            Output:
                z   :: torch.Tensor (same shape as input) - the output
                jac :: torch.Tensor (N,) - log Jacobian of output with respect to input
                reg :: torch.Tensor (M, 2) - regularization energies (Ek, Eg)
        '''
        rest_acc = None
        x = z[:, :, []]
        pz = z.shape[-1]
        for layer in reversed(self.layers):
            nz = prod(layer.res_shape)
            x = layer.merge(x, z[:, :, pz-nz:pz])
            pz -= nz
            bg = self.get_bg(layer, x.shape) # (N, type_dim, *shape)
            x, *rest = self.bijector.reverse(x, bg=bg, **kwargs)
            if rest_acc is None:
                rest_acc = rest
            else:
                rest_acc = tuple(a + b for a, b in zip(rest_acc, rest))
        if len(rest_acc) == 0:
            return x
        else:
            return x, *rest_acc

class RGModel(torch.nn.Module):
    ''' RG flow-based generative model
        
        Parameters:
            shape :: torch.Size - data shape
            dim :: int - feature dimensions
            type_dim :: int - type encoding dimension
            --- optional ---
            base_dist :: str - base distribution 'Normal' or 'Laplace'
            hdims :: list of int - hidden feature dimensions (for all hidden layers)
            ksize :: int - kernel size (should be odd)
            stride :: int - stride of partition
            bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
            bias :: bool - If True, adds a learnable bias to the output
            activation :: str - activation function name
            hyper_dim :: int - latent space dimension of hyper net
    '''
    def __init__(self, shape, dim, type_dim, base_dist='Normal', **kwargs):
        super().__init__()
        self.rgflow = RGFlow(shape, dim, type_dim, **kwargs)
        base_dist = getattr(torch.distributions, base_dist)(0., 1.)
        self.base_dist = base_dist.expand((dim,)+self.rgflow.out_shape)
    
    def extra_repr(self):
        return f'(base_dist): {self.base_dist}'
    
    def log_prob(self, x, jac=None, **kwargs):
        z, logJ, *rest = self.rgflow.encode(x, jac=True, **kwargs)
        logpz = self.base_dist.log_prob(z).view(z.shape[:1]+(-1,)).sum(-1)
        if len(rest) == 0:
            return logpz + logJ
        else:
            return logpz + logJ, *rest
    
    def sample(self, n_sample):
        ''' sample from the generative model
            Input:
                n_sample :: int - number of samples to generate
            Output
                x :: torch.Tensor (N, dim, *shape) - generated samples
        '''
        with torch.no_grad():
            return self.rsample(n_sample)

    def rsample(self, n_sample):
        ''' reparametrized sampling, enables gradient back propagation. 
            (see .sample) '''
        z = self.base_dist.rsample([n_sample])
        x = self.rgflow.decode(z)
        return x
    
    def lhmc_sampler(self, energy, x0, **kwargs):
        ''' construct a latent space HMC sampler 
            Input:
                energy :: func or nn.Module - energy function E(x)
                x0 :: torch.Tensor - initial configruation of x
            Output:
                lhmc_sampler :: LatentHMCSampler - LHMC sampler
                    providing sample method to perform HMC in latent space '''
        return LatentHMCSampler(self.rgflow, energy, x0, **kwargs)

    def nll_loss(self, x, lk=0.01, lg=0.01, **kwargs):
        ''' compute negative log likelihood loss given training samples
            Input:
                x :: torch.Tensor (N, dim, *shape) - training samples
                lk :: real - kinetic energy regularization strength
                lg :: real - gradient energy regularization strength
            Output:
                loss :: torch.Tensor (scalar) - regularized loss
                -logpx :: torch.Tensor (scalar) - negative log likelihood
                Ek :: torch.Tensor (scalar) - kinetic energy
                Eg :: torch.Tensor (scalar) - gradient energy
        '''
        logpx, reg = self.log_prob(x, reg=True, **kwargs)
        logpx = logpx.mean(0)
        reg = reg.mean(0)
        ls = torch.tensor([lk, lg])
        loss = -logpx + reg.dot(ls)
        return loss, -logpx, reg[0], reg[1]

    def free_loss(self, energy, n_sample, lk=0.01, lg=0.01, **kwargs):
        ''' compute free energy loss given energy function
            Input:
                energy :: func or nn.Module - energy function E(x)
                n_sample :: int - number of samples used to estimate free energy
                lk :: real - kinetic energy regularization strength
                lg :: real - gradient energy regularization strength
            Output:
                loss :: torch.Tensor (scalar) - regularized loss
                F :: torch.Tensor (scalar) - free energy
                Ek :: torch.Tensor (scalar) - kinetic energy
                Eg :: torch.Tensor (scalar) - gradient energy
        '''
        z = self.base_dist.rsample([n_sample])
        x, logJ, reg = self.rgflow.decode(z, jac=True, reg=True, **kwargs)
        F = (energy(x) - logJ).mean()
        reg = reg.mean(0)
        ls = torch.tensor([lk, lg])
        loss = F + reg.dot(ls)
        return loss, F, reg[0], reg[1]






