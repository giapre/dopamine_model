#!pip install vbjax
#See at DOI: 10.5281/zenodo.14204249

import time
import os
import collections
import jax
import jax.tree_util
import jax.numpy as jnp
import numpy as np
import numpy.random as nr
import vbjax as vb
import gast_model as gm

#import matplotlib.pyplot as plt
#from matplotlib.ticker import FuncFormatter

"""
Functions for computing the BOLD signal
"""

def make_run(init, tavg_period, total_time, dt, sigma, adhoc):
    '''returns the run function;
    so it basically serves as a factory function that creates
    and configures the run function based on the provided parameters.
    '''
    nt = int(tavg_period / dt) # chunk size
    nta = int(np.ceil(total_time / tavg_period)) # number of chunk

    noise_intensity = jnp.array([0., 0.1, 0.01, 0., 0., 0., 0.]).reshape(-1,1)*sigma
    _, loop = vb.make_sde(dt, gm.dopa_net, noise_intensity, adhoc)

    # setup tavg monitor
    n_nodes = init.shape[1]
    ta_buf, ta_step, ta_sample = vb.make_timeavg((init.shape[0], n_nodes))
    ta_sample = vb.make_offline(ta_step, ta_sample)

    # now setup bold
    bold_buf, bold_step, bold_samp = vb.make_bold(
        shape=init[0].shape,  # only r
        dt=tavg_period/1e3, #trick: 10*
        p=vb.bold_default_theta)

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
    Takes into argument the (global coupling, model parameters, SC matrix) and random generation key
    Returns the raw temporal average and the downsampled BOLD
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