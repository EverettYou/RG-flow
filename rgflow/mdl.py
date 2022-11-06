import torch
from .rg import RGFlow

from numbers import Number
from torch.distributions import constraints
class ShapedNormal(torch.distributions.Distribution):
    ''' Shaped normal distribution
        
        Parameters:
            event_shape :: torch.Size - event shape (dim, *shape)
            loc :: torch.Tensor - mean
            scale :: torch.Tensor - standard deviation
    '''
    arg_constraints = {}
    support = constraints.real
    has_rsample = True
    
    def  __init__(self, event_shape, loc=0., scale=1.):
        super(type(self), self).__init__(event_shape=torch.Size(event_shape))
        self.event_dim = len(self.event_shape)
        self.loc = self.cast(loc)
        self.scale = self.cast(scale)
        
    def __repr__(self):
        return self.__class__.__name__ + f'(event_shape: {tuple(self.event_shape)})'

    @property
    def dist(self):
        return torch.distributions.Normal(self.loc, self.scale)
    
    def new(self, loc=0., scale=1.):
        ''' returns a new instance with updated loc and scale,
            but the same event_shape '''
        return type(self)(self.event_shape, loc, scale)

    def to(self, *args, **kwargs):
        self.loc = self.loc.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        return self
        
    def cast(self, val):
        ''' cast val to (..., *event_shape) '''
        if isinstance(val, Number):
            return self.cast(torch.tensor(val))
        else:
            if val.dim() == 0:
                return val.expand(self.event_shape)
            elif val.shape[-1] == self.event_shape[0]:
                for n in self.event_shape[1:]:
                    val = val.unsqueeze(-1).expand(val.shape+(n,))
                return val
            elif val.dim() >= self.event_dim and val.shape[-self.event_dim:] == self.event_shape:
                return val
            else:
                raise RuntimeError(f'Can not cast val of shape {tuple(val.shape)} to the event shape {tuple(self.event_shape)}.')
    
    def rsample(self, samples):
        return self.dist.rsample([samples])
        
    def log_prob(self, value):
        ''' log probability (summed within even_shape) '''
        log_prob = self.dist.log_prob(value)
        return log_prob.view(*log_prob.shape[:-self.event_dim], -1).sum(-1)


class RGGenerator(torch.nn.Module):
    ''' RG flow-based generative model
        
        Parameters:
            shape :: torch.Size - data shape
            dim :: int - feature dimensions
    '''
    def __init__(self, shape, dim, **kwargs):
        super().__init__()
        self.rgflow = RGFlow(shape, dim, **kwargs)
        self.base_dist = ShapedNormal([dim, *self.rgflow.out_shape])
    
    def extra_repr(self):
        return f'(base_dist): {self.base_dist}'

    def to(self, *args, **kwargs):
        self.rgflow = self.rgflow.to(*args, **kwargs)
        self.base_dist = self.base_dist.to(*args, **kwargs)
        return self
    
    def log_prob(self, x, mode='jf', **kwargs):
        ''' log probability estimation 
            Input:
                x :: torch.Tensor (N, dim, *shape) - samples 
            Ouput:
                log p(x) :: torch.Tensor (N,) - log probability '''
        z, logJ, *rest = self.rgflow.encode(x, mode=mode, **kwargs)
        logpz = self.base_dist.log_prob(z)
        return logpz + logJ
    
    def sample(self, samples):
        ''' sample from the generative model
            Input:
                samples :: int - number N of samples to generate
            Output
                x :: torch.Tensor (N, dim, *shape) - generated samples
        '''
        with torch.no_grad():
            return self.rsample(samples)

    def rsample(self, samples):
        ''' reparametrized sampling, enables gradient back propagation. 
            (see .sample) '''
        z = self.base_dist.rsample(samples)
        x, *_ = self.rgflow.decode(z)
        return x
    
    def nll_loss(self, x, lk=0.01, lg=0.01, mode='jf_reg', **kwargs):
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
        z, logJ, Ek, Eg = self.rgflow.encode(x, mode=mode, **kwargs)
        logpz = self.base_dist.log_prob(z)
        logpx = logpz + logJ
        logpx, Ek, Eg = [val.mean() for val in (logpx, Ek, Eg)]
        return -logpx + lk * Ek + lg * Eg, -logpx, Ek, Eg

    def free_loss(self, energy, samples, lk=0.01, lg=0.01, mode='jf_reg', **kwargs):
        ''' compute free energy loss given energy function
            Input:
                energy :: func or nn.Module - energy function E(x)
                lk :: real - kinetic energy regularization strength
                lg :: real - gradient energy regularization strength
            Output:
                loss :: torch.Tensor (scalar) - regularized loss
                F :: torch.Tensor (scalar) - free energy
                Ek :: torch.Tensor (scalar) - kinetic energy
                Eg :: torch.Tensor (scalar) - gradient energy
        '''
        z = self.base_dist.rsample([samples])
        x, logJ, *rest = self.rgflow.decode(z, mode=mode, **kwargs)
        logpz = self.base_dist.log_prob(z)
        logpx = logpz - logJ
        F = (energy(x) + logpx).mean()
        Ek, Eg = [val.mean() for val in rest]
        return F + lk * Ek + lg * Eg, F, Ek, Eg



    

