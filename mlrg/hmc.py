import torch
from math import ceil

class HMCSampler():
    ''' Hamiltonian Monte Carlo sampler 
        
        Example:
        energy = lambda x: x.square().sum(-1)/2
        hmc = HMCSampler(energy, [3,2])
        hmc.sample()
        
        Parameters:
        energy :: func or torch.Module - energy function
        shape :: torch.Size or int ... - shape of sample (including batch size)
        --- optional ---
        step_size :: real - leap-frog step size
        adaptive :: bool - if True, adjust the step size adaptively to approach the target acception rate
        target_rate :: real - target acceptance rate
        shrink_factor :: real in (0.,1.) - the factor to shrink step size
        max_step_size :: real - step size upper bound
        min_step_size :: real - step size lower bound
        thermal_iters :: int - default iterations to thermalize
        '''
    def __init__(self, energy, shape,
                 step_size=0.02, 
                 adaptive=True, 
                 target_rate=0.65, 
                 shrink_factor=0.8, 
                 max_step_size=10000., 
                 min_step_size=0.0001,
                 thermal_iters=10):
        self.energy = energy
        self.step_size = step_size
        self.adaptive = adaptive
        self.target_rate = target_rate
        self.shrink_factor = shrink_factor
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size 
        self.thermal_iters = thermal_iters
        self.x = torch.zeros(shape)
        
    def grad_energy(self, x):
        x_requires_grad = x.requires_grad # record grad requirement
        with torch.enable_grad():
            x.requires_grad_(True)
            grad_energy = torch.autograd.grad(self.energy(x).sum(), x, create_graph=x_requires_grad)[0]
        x.requires_grad_(x_requires_grad) # reset grad requirement
        return grad_energy
    
    def leap_frog(self, x0, p0, step_size=None, traj_len=10, **kwargs):
        step_size = self.step_size if step_size is None else step_size
        # start by half-step momentum update
        p = p0 - 0.5 * step_size * self.grad_energy(x0)
        x = x0 + step_size * p
        for t in range(traj_len):
            # full step update
            p = p - step_size * self.grad_energy(x)
            x = x + step_size * p
        # end with half-step momentum update
        p =  p - 0.5 * step_size * self.grad_energy(x)
        return x, p
    
    def hamiltonian(self, x, p):
        V = self.energy(x)
        p = p.view(V.shape+(-1,))
        K = p.square().sum(-1) / 2
        return K + V
        
    def step(self, x0, **kwargs):
        p0 = torch.randn_like(x0)
        H0 = self.hamiltonian(x0, p0)
        x, p = self.leap_frog(x0, p0, **kwargs)
        H = self.hamiltonian(x, p)
        prob_accept = torch.exp(H0 - H)
        mask = prob_accept > torch.rand_like(prob_accept)
        shape = mask.shape + (1,) * (x.dim() - mask.dim())
        x = torch.where(mask.view(shape), x, x0)
        self.adjust_step_size(mask.float().mean())
        return x
    
    def adjust_step_size(self, accept_rate):
        if self.adaptive:
            if accept_rate > self.target_rate:
                new_step_size = self.step_size / self.shrink_factor
            else:
                new_step_size = self.step_size * self.shrink_factor
            self.step_size = max(min(new_step_size, self.max_step_size), self.min_step_size)
            
    def sample(self, x=None, iters=None, samples=None, requires_grad=False, **kwargs):
        ''' generate samples
            Input:
                x :: torch.Tensor - initial samples (optional)
                iters :: int - number of iterations (optional)
                samples :: int - number of samples (optional)
                    (samples will extend as necessary to adapt to the sample number)
                requires_grad :: bool - if True, will enable autograd recording 
                step_size :: real - step size (override the sampler's setting)
                traj_len :: int - leap frog trajectory length 
            Output:
                x :: torch.Tensor - generated sample '''
        x = self.x if x is None else x
        iters = self.thermal_iters if iters is None else iters
        x.requires_grad_(requires_grad)
        if samples is not None and samples != x.shape[0]:
            x = x.repeat((ceil(samples/x.shape[0]),)+(1,)*(x.dim()-1))[:samples]
        for _ in range(iters):
            x = self.step(x, **kwargs)
        self.x = x.detach()
        return x