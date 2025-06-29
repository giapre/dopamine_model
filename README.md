**SUMMARY**

In this work we show the use of a neural mass model equipped with D1 dopaminergic modulation of AMPA currents. This model is an extended version of the work of Gast et al (https://doi.org/10.1073/pnas.2311885121) that includes a dose-dependent neuromodulatory effect of dopamine, similarly to that of Depannemaecker et al (https://doi.org/10.1101/2024.06.23.600260). 

This model simulates the brain as a network of neural mass models, integrating dopamine dynamics with glutamatergic and GABAergic synaptic interactions. It adapts a neural mass model to include the effects of dopamine via the D1 receptor, which enhances excitatory synaptic currents (AMPA and NMDA channels). Dopaminergic inputs and reuptake regulate dopamine concentration at each node. 
The weights for these connections are derived from tractography data, adjusted based on connectivity studies. The whole-brain model can be implemented using three atlases, Desikan-Killiany (DK) and Schaefer 100 Parcels 7 Networks, and AAL2, with specific masks for glutamatergic, GABAergic, and dopaminergic projections. These projections are informed by established brain circuits using anatomical knowledge. D1 receptor numbers for brain regions are added using PET data, aligning receptor density values with each node in the model. 
This approach allows for the simulation of neuromodulatory effects across different brain areas, with flexibility in adjusting network dynamics based on different parameterizations.  

In this repository, the core mathematical model is implemented in gast_model.py, which includes equations for excitatory-only systems, as well as extended formulations that incorporate inhibitory and dopaminergic dynamics. Parameter values used in the equations are also provided within the same file.

The utils.py module contains utilities for simulating BOLD signals and computing Functional Connectivity Dynamics (FCD). They leverage on vbjax (Marmaduke Woodman, Abolfazl Ziaeemehr, see doi: 10.5281/zenodo.14204249), a library for numerical integration routines based on the JAX library. 

The example.ipynb notebook demonstrates how to use the model in both single-node and whole-brain simulations. It begins by analyzing the dynamical repertoire of a single node across different input currents, illustrating the range of regimes the system can express. It then moves to full-network simulations, showcasing how to compute BOLD signals, structure-function connectivity (SCFC), and FCD. Finally, the notebook includes a brief parameter sweep over the excitatory connectivity weight (we), highlighting an optimal range where SCFC and FCD are maximized.

The folder structural_data contains all the weights, connectivity masks, and receptor files for the three atlases.

**For running the model**
You will simply need to install vbjax in your python environment, and you can get started! You can check the versions you need in the notebook example.ipynb. 

**Please read the documentation.pdf file for a detailed description of the model and references.**
