import torch
from functorch import make_functional # https://pytorch.org/functorch/stable/
from torchdiffeq import odeint_adjoint as odeint # https://github.com/rtqichen/torchdiffeq
from math import prod

class CNN(torch.nn.Module):
    ''' Convolutional Neural Network (CNN)
        Constructs a (pooling-free) CNN for d-dimensional (d=1,2,3) data
    
    Example:
        x = torch.randn(1, 2, 6, 6)
        net = CNN(2, 2, [3, 5], ksize=3)
        net(x)
    
    Parameters:
        d :: int - data dimension (1D, 2D, 3D)
        dim :: int - visible features
        hdims :: list of int - hidden features (for all hidden layers)
        ksize :: int - kernel size (should be odd for padding to work properly)
        bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
        bias :: bool - If True, adds a learnable bias to the output
        activation :: str - activation function name
    '''
    def __init__(self, d, dim, hdims=[], ksize=3, bdy_cond='zeros', bias=True, activation='Tanh', **kwargs):
        assert ksize is None or ksize%2==1, 'ksize={} must be an odd integer'.format(ksize)
        super().__init__()
        try:
            conv = {1:torch.nn.Conv1d, 2:torch.nn.Conv2d, 3:torch.nn.Conv3d}[d]
        except:
            raise NotImplementedError(f'CNN with data dimension {d} is not implemented.')
        self.layers = torch.nn.ModuleList()
        dim_in, dim_out = dim, dim
        for dim_out in hdims:
            self.layers.append(conv(dim_in, dim_out, kernel_size=ksize, 
                    padding=(ksize-1)//2, padding_mode=bdy_cond, bias=bias))
            dim_in = dim_out
            self.layers.append(getattr(torch.nn, activation)())
        self.layers.append(conv(dim_out, dim, kernel_size=ksize, 
                padding=(ksize-1)//2, padding_mode=bdy_cond, bias=bias))
        
    def forward(self, x):
        ''' CNN forward 
            Input:
                x :: torch.Tensor (N, dim, ...) or (dim, ...)
                    N - batch size
                    dim - feature dimension
                    ... - L for 1D data 
                        - (H,W) for 2D data
                        - (D,H,W) for 3D data
            Ourput:
                cnn(x) :: torch.Tensor (same shape as input)
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    
class Dynamic(torch.nn.Module):
    ''' Converts a static base module to a time-dependent dynamic module.
        Given a base module that computes y = g(x|p) for input x and parameter p,
        this module construct a hypernet h(t) to model the time-dependent parameter,
        and compute y = f(t, x) = g(x|p=h(t))
        [code adapted from https://github.com/shyamsn97/hyper-nn/blob/main/hypernn/torch/utils.py]
        
        Example:
            t = torch.tensor(0.)
            x = torch.randn(4, 3)
            model = DynamicModule(torch.nn.Linear(3, 3), 5)
            model(t, x)
        
        Parameters:
            base_module :: torch.nn.Module - static base module
            hyper_dim :: int - latent space dimension of hyper net
    '''
    def __init__(self, base_module, hyper_dim=None, **kwargs):
        super().__init__()
        self.hyper_dim = hyper_dim
        if hyper_dim is None:
            self.base_module = base_module
        else:
            self.func_module, self.params = make_functional(base_module)
            # comput number of elements in the base layer
            numel = sum(param.numel() for param in self.params)
            # construct hyper network (input: time (scalar), output: parameter vector)
            self.hyper_net = torch.nn.Sequential(torch.nn.Linear(1, hyper_dim), 
                                                 torch.nn.Tanh(),
                                                 torch.nn.Linear(hyper_dim, numel))
    
    def forward(self, t, x):
        ''' DynamicModule forward f(t, x)
            Input:
                t :: torch.Tensor (scalar) - time
                x :: torch.Tensor - input data
            Output:
                f(t, x) :: torch.Tensor - output data
        '''
        if self.hyper_dim is None:
            return self.base_module(x)
        else:
            # infer parameters from t
            param_vec = self.hyper_net(t.view(1,1)).view(-1)
            # load parameters to the base layer
            params = []
            start = 0
            for param in self.params:
                end = start + param.numel()
                params.append(param_vec[start:end].view_as(param))
            return self.func_module(params, x)

class ODEFunc(torch.nn.Module):
    ''' ODE function that defines dx/dt = f(t,x).
        Other related ODEs:
            Jacobian: dlogJ/dt = div f(t, x)
            kinetic energy: dEk/dt = |f(t,x)|^2
            gradient energy: dEg/dt = |grad f(t,x)|^2
        [see neural ODE arXiv:1806.07366, 1810.01367, 2002.02798]

        Parameters:
            f :: DynamicModule - dynamic module that models f(t,x)
    '''
    div = None
    e = None
    def __init__(self, f):
        super().__init__()
        self.f = f
    
    def forward(self, t, state):
        ''' ODE function forward d(state)/dt = f(t, state)
            Input:
            t: torch scalar - time parameter
            x or (x, logJ) or (x, logJ, Ek, Eg): 
                x :: torch.Tensor (N, dim, ...)
                logJ :: torch.Tensor (N,) - log Jacobian
                Ek :: torch.Tensor (N,) - kinetic energy regulator
                Eg :: torch.Tensor (N,) - gradient energy regulator

            Output:
            dx or (dx, dlogJ) or (dx, dlogJ, dEk, dEg): 
                dx :: torch.Tensor (N, dim, ...)
                dlogJ :: torch.Tensor (N,)
                dEk :: torch.Tensor (N,)
                dEg :: torch.Tensor (N,)
        '''
        # ODE state can contain x, logJ, Ek, Eg  
        if isinstance(state, tuple):
            if len(state) == 1:   # state = (x,)
                return (self.f(t, state[0]),) # call ODE function f
            elif len(state) == 2: # state = (x, logJ)
                return self.jf(t, state) # call ODE function jf with dlogJ return
            elif len(state) == 4: # state = (x, logJ, Ek, Eg)
                return self.jf_reg(t, state) # call ODE function jf_reg with dlogJ and regularizations return
            else:
                raise NotImplementedError('ODEFunc can not handle {} state tensors.'.format(len(state)))
        else: # state = x
            return self.f(t, state) # call ODE function f
    
    # Jacobbian forward
    def jf(self, t, state):
        x = state[0] # state = (x, logJ)
        with torch.set_grad_enabled(True):
            # important to require grad for x and t, otherwise grad can not back prop
            x.requires_grad_(True)
            t.requires_grad_(True)
            dx = self.f(t, x)
            dlogJ, _ = self.div(dx, x, reg=False) # estimate div f, without estimating regulations
        return (dx, dlogJ)
    
    # Jacobbian forward with regularization [arXiv:2002.02798] 
    def jf_reg(self, t, state):
        x = state[0] # state = (x, logJ, Ek, Eg)
        with torch.set_grad_enabled(True):
            # important to require grad for x and t, otherwise grad can not back prop
            x.requires_grad_(True)
            t.requires_grad_(True)
            dx = self.f(t, x)
            dlogJ, dEg = self.div(dx, x, reg=True) # estimate div f and regulations
        dEk = (dx**2).view(dx.shape[0],-1).sum(-1)
        return (dx, dlogJ, dEk, dEg) 
    
    # to be called before odeint, to set div method and prepare Gaussian noise if needed
    def setup(self, x, div='approx'):
        if div == 'exact':
            self.div = self.div_exact
        elif div == 'approx': # approximate method need to use Gaussian noise
            self.div = self.div_approx
            self.e = torch.randn_like(x) # set Gaussian noise (DONT use rand_like)
        else:
            raise NotImplementedError('Unknown method {} for divergence estimation.'.format(div))
    
    # divegence estimators
    # exact method (slow and memory intensive, used for benchmarking only)
    def div_exact(self, dx, x, reg=False):
        div = 0.   # divergence (for dlogJ)
        grad2 = 0. # square gradient (for dEg)
        dx = dx.view(dx.shape[0],-1) # assuming dx:(N,...), N - batch dim
        for i in range(dx.shape[-1]): # loop through all variable in ...
            # autograd to compute d(dx_i)/dx, same shape as x
            ddxi = torch.autograd.grad(dx[:, i].sum(), x, create_graph=True)[0]
            div += ddxi.view(x.shape[0],-1)[:, i].contiguous() # accumulate divergence
            if reg: # if regularization is required
                grad2 += (ddxi**2).view(x.shape[0],-1).sum(-1) # also accumulate square gradient
        return div, grad2
        
    # approximate method [FFJORD arXiv:1810.01367]
    # div f = avg_e d(e.dx)/d(e.x) averge over Gaussian random noise e
    def div_approx(self, dx, x, reg=False):
        assert self.e.shape == x.shape, 'noise shape e:{} and input shape x:{} not compatible.'.format(self.e.shape, x.shape)
        # eJ = d(e.dx)/dx, same shape of x
        eJ = torch.autograd.grad((dx * self.e).sum(), x, create_graph=True)[0]
        # div = eJ.e = d(e.dx)/d(e.x) -> sum_i d(dx_i)/d(x_i)
        div = (eJ * self.e).view(x.shape[0], -1).sum(-1) 
        # grad2 = (eJ)^2 = (d(e.dx)/dx)^2 -> sum_{i,j} (d(dx_i)/d(x_j))^2
        grad2 = (eJ**2).view(x.shape[0], -1).sum(-1) if reg else 0.
        return div, grad2

class ODEBijector(torch.nn.Module):
    ''' ODEBijector realizes a bijective map by solving the ODE dx/dt = f(t,x).

        Parameters:
            f :: DynamicModule - dynamic module that models f(t,x)
    '''
    def __init__(self, f):
        super().__init__()
        self.ode = ODEFunc(f)
    
    def forward(self, x, t0, t1, mode='f', div='approx', **kwargs):
        ''' ODE evolve x from t0 to t1

            Input:
                x :: torch.Tensor (N, dim, ...) - input tensor
                t0, t1 :: real or torch.Tensor (scalar) - starting and ending times
                mode :: str - computation mode
                    'f' : y = forward(x, t0, t1)
                    'jf' : (y, logJ) = forward(x, t0, t1)
                    'jf_reg' : (y, logJ, Ek, Eg) = forward(x, t0, t1)
                div :: str - method for divergence estimation
                    (this is only used if mode is 'jf' or 'jf_reg')
                    'approx' : approximate method
                    'exact' : exact method
            Output:
                y :: torch.Tensor (N, dim, ...) - output tensor
                logJ :: torch.Tensor (N,) - log Jacobian accumulated
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        ts = torch.tensor([t0, t1]).to(x)
        if mode == 'f':
            xs = odeint(self.ode, x, ts, **kwargs) # ode integration
            return (xs[-1],)
        else:
            zero = torch.zeros(x.shape[0]).to(x) # initial tensor for logJ, Ek, Eg
            # use the len of state to indicate regularization or not
            # 'jf': state = (x,0)
            # 'jf_reg': state = (x,0,0,0)
            state = (x,) + (zero,)*(1 if mode == 'jf' else 3) 
            self.ode.setup(x, div=div) # setup div method and noise befor odeint
            state = odeint(self.ode, state, ts, **kwargs) # ode integration
            return tuple(x[-1] if i < 2 else x[-1]/(t1-t0) for i, x in enumerate(state))

class RGPartition(torch.nn.Module):
    ''' partition fine-grained features to coarse-grained and residual features
        
        Parameters:
            in_shape :: torch.Size - data shape of fine-grained features
            stride :: int - stride of partition
    '''
    def __init__(self, in_shape, stride=2, **kwargs):
        assert stride>=2, f'stride must be greater equal than 2, but got {stride}.'
        super().__init__()
        self.in_shape = torch.Size(in_shape)
        out_shape = []
        mask = torch.tensor([True])
        for size in in_shape:
            axial_mask = torch.arange(size)%stride==stride//2
            mask = torch.kron(mask, axial_mask)
            out_shape.append(torch.count_nonzero(axial_mask))
        self.out_shape = torch.Size(out_shape)
        self.register_buffer('out_indx', torch.nonzero(mask).view(-1))
        self.register_buffer('res_indx', torch.nonzero(torch.logical_not(mask)).view(-1))
        self.res_shape = self.res_indx.shape
            
    def extra_repr(self):
        return f'{tuple(self.in_shape)} -> {tuple(self.out_shape)}, {tuple(self.res_shape)}'
        
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
    ''' perform one layer of RG transformation
        
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
        self.bijector = ODEBijector(Dynamic(CNN(len(in_shape), dim, **kwargs), **kwargs))
            
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
                logJ :: torch.Tensor (N,) - log Jacobian accumulated
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
                logJ :: torch.Tensor (N,) - log Jacobian accumulated
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        x = self.partitioner.merge(x, z) # merge data
        x, *rest = self.bijector(x, 1., 0., **kwargs) # bijector backward
        return x, *rest

class RGFlow(torch.nn.Module):
    ''' perform the RG flow
        
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
            shape = layer.partitioner.out_shape
    
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
                logJ :: torch.Tensor (N,) - log Jacobian accumulated
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
                logJ :: torch.Tensor (N,) - log Jacobian accumulated
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        rest_acc = None
        x = z[:, :, []]
        pz = z.shape[-1]
        for layer in reversed(self.layers):
            nz = prod(layer.partitioner.res_shape)
            x, *rest = layer.decode(x, z[:, :, pz-nz:pz], **kwargs)
            pz -= nz
            if rest_acc is None:
                rest_acc = rest
            else:
                rest_acc = tuple(R+r for R, r in zip(rest_acc, rest))
        return x, *rest_acc

class RGModel(torch.nn.Module):
    ''' RG flow-based generative model
        
        Parameters:
            shape :: torch.Size - data shape
            dim :: int - feature dimensions
            --- optional ---
            base_dist :: str - base distribution 'Normal' or 'Laplace'
    '''
    def __init__(self, shape, dim, base_dist='Normal', **kwargs):
        super().__init__()
        self.rgflow = RGFlow(shape, dim, **kwargs)
        base_dist = getattr(torch.distributions, base_dist)(0., 1.)
        self.base_dist = base_dist.expand((dim,)+self.rgflow.out_shape)
    
    def extra_repr(self):
        return f'(base_dist): {self.base_dist}'
    
    def log_prob(self, x, mode='jf', **kwargs):
        z, logJ, *rest = self.rgflow.encode(x, mode=mode, **kwargs)
        logpz = self.base_dist.log_prob(z).view(z.shape[:1]+(-1,)).sum(-1)
        return logpz + logJ, *rest
    
    def sample(self, samples):
        ''' sample from the generative model
            Input:
                samples :: int - number of samples to generate
            Output
                x :: torch.Tensor (N, dim, *shape) - generated samples
        '''
        with torch.no_grad():
            return self.rsample([samples])

    def rsample(self, samples):
        ''' reparametrized sampling, enables gradient back propagation. 
            (see .sample) '''
        z = self.base_dist.rsample([samples])
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
        vals = self.log_prob(x, mode=mode, **kwargs)
        logpx, Ek, Eg = [val.mean() for val in vals]
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
        F = (energy(x) - logJ).mean()
        Ek, Eg = [val.mean() for val in rest]
        return F + lk * Ek + lg * Eg, F, Ek, Eg






    