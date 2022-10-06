import torch
from torchdiffeq import odeint_adjoint as odeint # https://github.com/rtqichen/torchdiffeq

class ODEFunc(torch.nn.Module):
    ''' Given the function f(t,x) that defines the ODE dx/dt = f(t,x),
        ODEFunc computes the other related ODEs:
            Jacobian: dlogJ/dt = div f(t,x)
            kinetic energy: dEk/dt = |f(t,x)|^2
            gradient energy: dEg/dt = |grad f(t,x)|^2
        [see neural ODE arXiv:1806.07366, 1810.01367, 2002.02798]

        Parameters:
            f :: Dynamic - dynamic module that models f(t,x)
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
                logJ :: torch.Tensor (N,) - accumulated log Jacobian
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
        ODE solver is implemented by calling torchdiffeq.odeint_adjoint as odeint.
        
        Example:
        ode = ODEBijector(Dynamic(torch.nn.Linear(2,2)))
        x0 = torch.randn(1,2)
        x1, logJ01, Ek01, Eg01 = ode(x0, 0., 1., mode='jf_reg', div='exact')
        x0_, logJ10, Ek10, Eg10 = ode(x1, 1., 0., mode='jf_reg', div='exact')
        (x0, x1, x0_), (logJ01, logJ10), (Ek01, Ek10), (Eg01, Eg10)

        Parameters:
            f :: Dynamic - dynamic module that models f(t,x)
    '''
    def __init__(self, f):
        super().__init__()
        self.odefunc = ODEFunc(f)
    
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
                logJ :: torch.Tensor (N,) - accumulated log Jacobian (log det dx(t1)/dx(t0))
                Ek :: torch.Tensor (N,) - accumulated kinetic energy
                Eg :: torch.Tensor (N,) - accumulated gradient energy
        '''
        ts = torch.tensor([t0, t1]).to(x)
        if mode == 'f':
            xs = odeint(self.odefunc, x, ts, **kwargs) # ode integration
            return (xs[-1],)
        else:
            zero = torch.zeros(x.shape[0]).to(x) # initial tensor for logJ, Ek, Eg
            # use the len of state to indicate regularization or not
            # 'jf': state = (x,0)
            # 'jf_reg': state = (x,0,0,0)
            state = (x,) + (zero,)*(1 if mode == 'jf' else 3) 
            self.odefunc.setup(x, div=div) # setup div method and noise befor odeint
            state = odeint(self.odefunc, state, ts, **kwargs) # ode integration
            return tuple(x[-1] if i < 2 else x[-1]/(t1-t0) for i, x in enumerate(state))

