import torch
from functorch import make_functional # https://pytorch.org/functorch/stable/

class CNN(torch.nn.Module):
    ''' Convolutional Neural Network (CNN)
        Constructs a (pooling-free) CNN for d-dimensional (d=1,2,3) data
    
    Example:
        x = torch.randn(1, 2, 6, 6)
        net = CNN(2, 2, [3, 5], ksize=3)
        net(x)
    
    Parameters:
        d :: int - data dimension (1D, 2D, 3D)
        dims :: list of int - feature dimensions (from input to hidden to output)
        ksize :: int - kernel size (should be odd for padding to work properly)
        bdy_cond :: str - boundary condition 'zeros', 'reflect', 'replicate' or 'circular'
        bias :: bool - If True, adds a learnable bias to the output
        activation :: str - activation function name
    '''
    def __init__(self, d, dims=[], ksize=3, bdy_cond='zeros', bias=True, activation='Tanh', **kwargs):
        assert ksize is None or ksize%2==1, 'ksize={} must be an odd integer'.format(ksize)
        super().__init__()
        try:
            conv = {1:torch.nn.Conv1d, 2:torch.nn.Conv2d, 3:torch.nn.Conv3d}[d]
        except:
            raise NotImplementedError(f'CNN with data dimension {d} is not implemented.')
        self.layers = torch.nn.ModuleList() # prepare empty module list
        dim_in = dims[0]
        for dim_out in dims[1:]:
            self.layers.append(conv(dim_in, dim_out, kernel_size=ksize, 
                    padding=(ksize-1)//2, padding_mode=bdy_cond, bias=bias)) # add conv layer
            dim_in = dim_out
            self.layers.append(getattr(torch.nn, activation)()) # add activation layer
        del self.layers[-1] # remove the last activation layer
        
    def forward(self, x):
        ''' CNN forward 
            Input:
                x :: torch.Tensor (N, dims[0], ...) or (dims[0], ...)
                    N - batch size
                    dim - feature dimension
                    ... - L for 1D data 
                        - (H,W) for 2D data
                        - (D,H,W) for 3D data
            Ourput:
                cnn(x) :: torch.Tensor (N, dims[-1], ...)
        '''
        for layer in self.layers:
            x = layer(x)
        return x
    
class Dynamic(torch.nn.Module):
    ''' Converts a static base module to a time-dependent dynamic module.
        Given a base module that computes v = g(x|p) for input x and parameter p,
        this module construct a hypernet h(t) to model the time-dependent parameter,
        and compute v = f(t, x) = g(x|p=h(t))
        [code adapted from https://github.com/shyamsn97/hyper-nn/blob/main/hypernn/torch/utils.py]
        
        Example:
            t = torch.tensor(0.)
            x = torch.randn(4, 3)
            model = Dynamic(torch.nn.Linear(3, 3), 5)
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
