#!pip install vbjax
#import vbjax as vb
#DOI: 10.5281/zenodo.14204249

import time
import os
import collections
import jax
import jax.tree_util
import jax.numpy as jnp
import numpy as np
import numpy.random as nr

import gast_model as gm

#import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter

"""
Functions for building time stepping loops.

"""

zero = 0
tmap = jax.tree_util.tree_map

def heun_step(x, dfun, dt, *args, add=zero, adhoc=None, return_euler=False):
    """Use a Heun scheme to step state with a right hand sides dfun(.)
    and additional forcing term add.
    """
    adhoc = adhoc or (lambda x,*args: x)
    d1 = dfun(x, *args)
    if add is not zero:
        xi = tmap(lambda x,d,a: x + dt*d + a, x, d1, add)
    else:
        xi = tmap(lambda x,d: x + dt*d, x, d1)
    xi = adhoc(xi, *args)
    d2 = dfun(xi, *args)
    if add is not zero:
        nx = tmap(lambda x,d1,d2,a: x + dt*0.5*(d1 + d2) + a, x, d1, d2, add)
    else:
        nx = tmap(lambda x,d1,d2: x + dt*0.5*(d1 + d2), x, d1, d2)
    nx = adhoc(nx, *args)
    if return_euler:
        return xi, nx
    return nx


def _compute_noise(gfun, x, p, sqrt_dt, z_t):
    g = gfun(x, p)
    try: # maybe g & z_t are just arrays
        noise = g * sqrt_dt * z_t
    except TypeError: # one of them is a pytree
        if isinstance(g, float): # z_t is a pytree, g is a scalar
            noise = tmap(lambda z: g * sqrt_dt * z, z_t)
        # otherwise, both must be pytrees and they must match
        elif not jax.tree_util.tree_all(jax.tree_util.tree_structure(g) == 
                                        jax.tree_util.tree_structure(z_t)):
            raise ValueError("gfun and z_t must have the same pytree structure.")
        else:
            noise = tmap(lambda g,z: g * sqrt_dt * z, g, z_t)
    return noise

def make_sde(dt, dfun, gfun, adhoc=None, return_euler=False, unroll=10):
    """Use a stochastic Heun scheme to integrate autonomous stochastic
    differential equations (SDEs).

    Parameters
    ==========
    dt : float
        Time step
    dfun : function
        Function of the form `dfun(x, p)` that computes drift coefficients of
        the stochastic differential equation.
    gfun : function or float
        Function of the form `gfun(x, p)` that computes diffusion coefficients
        of the stochastic differential equation. If a numerical value is
        provided, this is used as a constant diffusion coefficient for additive
        linear SDE.
    adhoc : function or None
        Function of the form `f(x, p)` that allows making adhoc corrections
        to states after a step.
    return_euler: bool, default False
        Return solution with local Euler estimates.
    unroll: int, default 10
        Force unrolls the time stepping loop.

    Returns
    =======
    step : function
        Function of the form `step(x, z_t, p)` that takes one step in time
        according to the Heun scheme.
    loop : function
        Function of the form `loop(x0, zs, p)` that iteratively calls `step`
        for all `z`.

    Notes
    =====

    In both cases, a Jax compatible parameter set `p` is provided, either an array
    or some pytree compatible structure.

    Note that the integrator does not sample normally distributed noise, so this
    must be provided by the user.


    >>> import vbjax as vb
    >>> _, sde = vb.make_sde(1.0, lambda x, p: -x, 0.1)
    >>> sde(1.0, vb.randn(4), None)
    Array([ 0.5093468 ,  0.30794007,  0.07600437, -0.03876263], dtype=float32)

    """

    sqrt_dt = jnp.sqrt(dt)

    # gfun is a numerical value or a function f(x,p) -> sig
    if not hasattr(gfun, '__call__'):
        sig = gfun
        gfun = lambda *_: sig

    def step(x, z_t, p):
        noise = _compute_noise(gfun, x, p, sqrt_dt, z_t)
        return heun_step(
            x, dfun, dt, p, add=noise, adhoc=adhoc,
            return_euler=return_euler)

    @jax.jit
    def loop(x0, zs, p):
        def op(x, z):
            x = step(x, z, p)
            # XXX gets unwieldy, how to improve?
            if return_euler:
                ex, x = x
            else:
                ex = None
            return x, (ex, x)
        _, xs = jax.lax.scan(op, x0, zs, unroll=unroll)
        if not return_euler:
            _, xs = xs
        return xs

    return step, loop

def make_offline(step_fn, sample_fn, *args):
    "Compute monitor samples in an offline or batch fashion."
    def op(mon, x):
        mon = step_fn(mon, x)
        return mon, None
    def offline_sample(mon, xs):
        mon, _ = jax.lax.scan(op, mon, xs)
        mon, samp = sample_fn(mon)
        return mon, samp
    return offline_sample

# NB shape here is the input shape of neural activity

def make_timeavg(shape):
    "Make a time average monitor."
    new = lambda : {'y': jnp.zeros(shape), 'n': 0}
    def step(buf, x):
        return {'y': buf['y'] + x,
                'n': buf['n'] + 1}
    def sample(buf):
        return new(), buf['y'] / buf['n']
    return new(), step, sample


def make_gain(gain, shape=None):
    "Make a gain-matrix monitor suitable for sEEG, EEG & MEG."
    tavg_shape = gain.shape[:1] + (shape[1:] if shape else ())
    buf, tavg_step, tavg_sample = make_timeavg(tavg_shape)
    step = lambda b, x: tavg_step(b, gain @ x)
    return buf, step, tavg_sample


# Bold implementation 

BOLDTheta = collections.namedtuple(
    typename='BOLDTheta',
    field_names='tau_s,tau_f,tau_o,alpha,te,v0,e0,epsilon,nu_0,'
                'r_0,recip_tau_s,recip_tau_f,recip_tau_o,recip_alpha,'
                'recip_e0,k1,k2,k3'
)

def compute_bold_theta(
        tau_s=0.65,
        tau_f=0.41,
        tau_o=0.98,
        alpha=0.32,
        te=0.04,
        v0=4.0,
        e0=0.4,
        epsilon=0.5,
        nu_0=40.3,
        r_0=25.0,
    ):
    recip_tau_s = 1.0 / tau_s
    recip_tau_f = 1.0 / tau_f
    recip_tau_o = 1.0 / tau_o
    recip_alpha = 1.0 / alpha
    recip_e0 = 1.0 / e0
    k1 = 4.3 * nu_0 * e0 * te
    k2 = epsilon * r_0 * e0 * te
    k3 = 1.0 - epsilon
    return BOLDTheta(**locals())

bold_default_theta = compute_bold_theta()

def bold_dfun(sfvq, x, p: BOLDTheta):
    s, f, v, q = sfvq
    ds = x - p.recip_tau_s * s - p.recip_tau_f * (f - 1)
    df = s
    dv = p.recip_tau_o * (f - v ** p.recip_alpha)
    dq = p.recip_tau_o * (f * (1 - (1 - p.e0) ** (1 / f)) * p.recip_e0
                          - v ** p.recip_alpha * (q / v))
    return jnp.array([ds, df, dv, dq])

def make_bold(shape, dt, p: BOLDTheta):
    "Make a BOLD fMRI monitor."
    sfvq = jnp.ones((4,) + shape)
    sfvq = sfvq.at[0].set(0)
    def step(sfvq, x):
        return heun_step(sfvq, bold_dfun, dt, x, p)
    def sample(buf):
        s, f, v, q = buf
        return buf, p.v0 * (p.k1*(1 - q) + p.k2*(1 - q / v) + p.k3*(1 - v))
    return sfvq, step, sample

"""
Functions for computing the BOLD signal
"""

# the `make_run` function just needs dimensions of the problem 
# the `make_run` function just needs dimensions of the problem 
def make_run(init, tavg_period, total_time, dt, sigma, adhoc):
    '''returns the run function;
    so it basically serves as a factory function that creates
    and configures the run function based on the provided parameters.
    '''
    nt = int(tavg_period / dt) # chunk size
    nta = int(np.ceil(total_time / tavg_period)) # number of chunk

    noise_intensity = jnp.array([0., 0.1, 0.01, 0., 0., 0., 0.]).reshape(-1,1)*sigma
    _, loop = make_sde(dt, gm.dopa_net, noise_intensity, adhoc)

    # setup tavg monitor
    n_nodes = init.shape[1]
    ta_buf, ta_step, ta_sample = make_timeavg((init.shape[0], n_nodes))
    ta_sample = make_offline(ta_step, ta_sample)

    # now setup bold
    bold_buf, bold_step, bold_samp = make_bold(
        shape=init[0].shape,  # only r
        dt=tavg_period/1e3, #trick: 10*
        p=bold_default_theta)

    # run function actually does the simulation based on inputs
    # that we might want to sweep over or change
    @jax.jit
    def run(params, key=jax.random.PRNGKey(42)):
 
        sim = {
            'ta': ta_buf,
            'bold': bold_buf,
            'init': init,
            'p': params,
            'key': key,
        }

        def sim_step(sim, t_key):
            t, key = t_key
            
            # sim['key'], key = jax.random.split(sim['key'])

            # generate randn and run simulation from initial conditions
            dw = jax.random.normal(key, (nt, init.shape[0], n_nodes))
            raw = loop(sim['init'], dw, sim['p'])

            # monitor results
            sim['ta'], ta_y = ta_sample(sim['ta'], raw)
            sim['bold'] = bold_step(sim['bold'], ta_y[0])
            _, bold_t = bold_samp(sim['bold'])
            sim['init'] = raw[-1]
            return sim, (ta_y, bold_t)

        ts = np.r_[:nta]*tavg_period
        #ts *= 10 #trick
        keys = jax.random.split(key, ts.size)
        sim, (ta_y, bold) = jax.lax.scan(sim_step, sim, (ts, keys))
        
        return ts, ta_y, bold
        
    return run

def run_for_parameters(p, key=jax.random.PRNGKey(0), plot=False) :
    
    """
    Takes into argument the (global coupling, MPR parameters, SC matrix) and random generation key
    Returns the downsampled BOLD, FC, and FCD
    """
    start = time.time()

    Ci, Ce, Cd, theta, sigma = p
    n_nodes = Ce.shape[0]
    init = jnp.array([0.01, -50.0, 0., 0.04, 0., 0., 0.,]) # Inizializzazione of the variables
    init = jnp.outer(init, jnp.ones(n_nodes))

    #you can adapt tavg_period ; #5 minutes = 300e3 ms
    #with the trick, we can do total_time=30e3 instead of 300e3 
    tavg_period=1; total_time=300e3; dt=5e-1
    run = make_run(init, tavg_period=tavg_period,  # 5ms or 200Hz
                        total_time=total_time, dt=dt, sigma=sigma,
                        adhoc=gm.dopa_stay_positive) 
    ts, ta_y, bold = run((Ci, Ce, Cd, theta), key)
    
    end = time.time() - start
    print('Simulating BOLD took ', end, ' seconds')
    
    #here we cut and downsample the bold to make it 296 data points
    #if not just write bold instead of bold_ds
    cut = 10000
    bold_ds = bold[cut::len(bold[cut:])//300][:296]
    #bold_ds = bold[cut:]

    return ta_y, bold_ds

def compute_fcd(ts, window_length=20, overlap=19):
    n_samples, n_regions = ts.shape
    #    if n_samples < n_regions:
    #        print('ts transposed')
    #        ts=ts.T
    #        n_samples, n_regions = ts.shape

    window_steps_size = window_length - overlap
    n_windows = int(np.floor((n_samples - window_length) / window_steps_size + 1))

    # upper triangle indices
    Isupdiag = np.triu_indices(n_regions, 1)    

    #compute FC for each window
    FC_t = np.zeros((int(n_regions*(n_regions-1)/2),n_windows))
    for i in range(n_windows):
        FCtemp = np.corrcoef(ts[window_steps_size*i:window_length+window_steps_size*i,:].T)
        FCtemp = np.nan_to_num(FCtemp, nan=0)
        FC_t[:,i] = FCtemp[Isupdiag]


    # compute FCD by correlating the FCs with each other
    FCD = np.corrcoef(FC_t.T)

    return FCD