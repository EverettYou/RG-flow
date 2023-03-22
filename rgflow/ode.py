import torch
from torchdiffeq import odeint_adjoint as odeint # https://github.com/rtqichen/torchdiffeq

class ODEFunc(torch.nn.Module):
    ''' Given the function f(t,x|y) that defines the ODE dx/dt = f(t,x|y),
        ODEFunc computes the other related ODEs:
            Jacobian: dlogJ/dt = div_x f(t,x|y)
            kinetic energy: dEk/dt = |f(t,x|y)|^2
            gradient energy: dEg/dt = |grad_x f(t,x|y)|^2
        [see neural ODE arXiv:1806.07366, 1810.01367, 2002.02798]

        Parameters:
            func :: torch.nn.Module - dynamic module that models f(t,x|y)
            div :: str - method for divergence estimation
                    (this is only used if jac or reg is True)
                    'approx' : approximate method
                    'exact' : exact method
            bg :: torch.Tensor - additional background tensor y to be concatenated with input x
            jac :: bool - whether to calculate differential Jacobian 
            reg :: bool - whether to calculate differential energies
    '''
    def __init__(self, func, div='approx', bg=None, jac=False, reg=False, **kwargs):
        super().__init__()
        self.func = func
        if div == 'exact':
            self.div = self.div_exact
        elif div == 'approx': # approximate method need to use Gaussian noise
            self.div = self.div_approx
            self.e = None
        else:
            raise NotImplementedError('Unknown method {} for divergence estimation.'.format(div))
        self.bg = bg
        self.jac = jac
        self.reg = reg
        
    def forward(self, t, state):
        out = []
        if isinstance(state, tuple):
            x = state[0]
        elif isinstance(state, torch.Tensor):
            x = state
        else:
            raise TypeError('Argument state must be a tensor or a tuple of tensors.')
        requires_grad = self.jac or self.reg
        with torch.set_grad_enabled(requires_grad):
            # require grad for x and t, otherwise grad can not back prop
            if requires_grad:
                x.requires_grad_(True)
                t.requires_grad_(True)
            if self.bg is None:
                dx = self.func(t, x)
            else:
                xbg = torch.cat([x, self.bg], axis = 1)
                dx = self.func(t, xbg)
            out.append(dx)
            if self.reg:
                dlogJ, dEg = self.div(dx, x, reg=True)
            else:
                if self.jac:
                    dlogJ = self.div(dx, x, reg=False)
        if self.jac:
            out.append(dlogJ)
        if self.reg:
            dEk = (dx**2).view(dx.shape[0],-1).sum(-1)
            out.append(torch.stack([dEk, dEg], axis=-1))
        return tuple(out)
        
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
        if reg:
            return div, grad2
        else:
            return div
        
    # approximate method [FFJORD arXiv:1810.01367]
    # div f = avg_e d(e.dx)/d(e.x) averge over Gaussian random noise e
    def div_approx(self, dx, x, reg=False):
        if self.e is None:
            self.e = torch.randn_like(x) # set Gaussian noise (DONT use rand_like)
        # eJ = d(e.dx)/dx, same shape of x
        eJ = torch.autograd.grad((dx * self.e).sum(), x, create_graph=True)[0]
        # div = eJ.e = d(e.dx)/d(e.x) -> sum_i d(dx_i)/d(x_i)
        div = (eJ * self.e).view(x.shape[0], -1).sum(-1) 
        if reg:
            # grad2 = (eJ)^2 = (d(e.dx)/dx)^2 -> sum_{i,j} (d(dx_i)/d(x_j))^2
            grad2 = (eJ**2).view(x.shape[0], -1).sum(-1)
            return div, grad2
        else:
            return div

class ODEBijector(torch.nn.Module):
    ''' ODEBijector realizes a bijective map by solving the ODE dx/dt = f(t,x|y)
        from time 0 to 1, given y as some extra background tensor (like position encoding).
        ODE solver is implemented by calling torchdiffeq.odeint_adjoint as odeint.

        Parameters:
            func :: torch.nn.Module - dynamic module that models f(t,x|y),
                    the return of f(t,x|y) must be the same shape as x,
                    x|y will be passed to f as concatenated features.       
    '''
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def state(self, x, jac=False, reg=False, **kwargs):
        state = [x]
        if jac:
            state.append(torch.zeros((x.shape[0])).to(x))
        if reg:
            state.append(torch.zeros((x.shape[0], 2)).to(x))
        return tuple(state)
        
    def result(self, states, jac=False, reg=False, **kwargs):
        states_iter = iter(states)
        result = [next(states_iter)[-1]]
        if jac:
            result.append(next(states_iter)[-1])
        if reg:
            # regularizations are assumed to be positive regardless forward/reverse 
            result.append(next(states_iter)[-1].abs())
        return tuple(result)
        
    def solve(self, x, t0, t1, **kwargs):
        ''' solve ODE to evolve x from t0 to t1

            Input:
                x :: torch.Tensor (N, dim, ...) - input tensor
                t0, t1 :: real or torch.Tensor (scalar) - starting and ending times
                div :: str - method for divergence estimation
                    (this is only used if jac or reg is True)
                    'approx' : approximate method
                    'exact' : exact method
                bg :: torch.Tensor - additional background tensor to be concatenated with input
                jac :: bool - whether to calculate differential Jacobian 
                reg :: bool - whether to calculate differential energies
                --- ODE solver parameters ---
                rtol: upper bound on relative error
                atol: upper bound on absolute error
                method: integration method, default 'dopri5'
                (see https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/odeint.py)
                
            Output:
                x   :: torch.Tensor (same shape as input) - the output
                jac :: torch.Tensor (N,) - log Jacobian of output with respect to input
                reg :: torch.Tensor (M, 2) - regularization energies (Ek, Eg)
        '''
        ts = torch.tensor([t0, t1], device=x.device)
        state = self.state(x, **kwargs)
        odefunc = ODEFunc(self.func, **kwargs)
        odekwargs = {key: value for key, value in kwargs.items() if key in {'rtol', 'atol', 'method', 'options', 'event_fn'}}
        states = odeint(odefunc, state, ts, **odekwargs)
        return self.result(states, **kwargs)
        
    def forward(self, x, **kwargs):
        ''' Forward map of the bijector
            (see solve method) '''
        return self.solve(x, 0., 1., **kwargs)
        
    def reverse(self, x, **kwargs):
        ''' Backward map of the bijector
            (see solve method) '''
        return self.solve(x, 1., 0., **kwargs)
