**SUMMARY**

This model simulates the brain as a network of neural mass models, integrating dopamine dynamics with glutamatergic and GABAergic synaptic interactions. It adapts a neural mass model to include the effects of dopamine via the D1 receptor, which enhances excitatory synaptic currents (AMPA and NMDA channels). Dopaminergic inputs and reuptake regulate dopamine concentration at each node. 
The weights for these connections are derived from tractography data, adjusted based on connectivity studies. The whole-brain model can be implemented using two atlases, Desikan-Killiany (DK) and Schaefer 7 Networks, with specific masks for glutamatergic, GABAergic, and dopaminergic projections. These projections are informed by established brain circuits using anatomical knowledge. D1 receptor numbers for brain regions are added using PET data, aligning receptor density values with each node in the model. 
This approach allows for the simulation of neuromodulatory effects across different brain areas, with flexibility in adjusting network dynamics based on different parametrization. 

Simulations are implemented with functions taken from vbjax, a Jax-based package for working with virtual brain style models (1).

**MODEL DOCUMENTATION**

Impairments of the neuromodulatory system have been considered key pathophysiological mechanisms of many neurological and psychiatric diseases, such as Parkinson's disease and schizophrenia. 

We aim to construct a model of the brain as a network of neural mass models extended to include dopamine concentration and its effects, along with its interplay with glutamatergic and GABAergic synapses. This can be achieved by 
1.	embedding a neural mass model with dopaminergic neuromodulatory mechanisms, 
2.	creating a brain network that accounts for the interactions of glutamatergic, GABAergic, and dopaminergic projections between its nodes.

Figure 1. The brain is modeled as a network of interconnected nodes, each governed by a neural mass model. Nodes exchange signals through three types of inputs: glutamatergic, GABAergic, and dopaminergic. Glutamatergic inputs excite the node, GABAergic inputs inhibit it, and dopaminergic inputs modulate its excitatory drive. 
Model equations

The neural mass model presented here is an adapted version of the work of Gast et al (2) to account for the generic dynamics of neuromodulators, similarly to what has been proposed by Depannemaecker et al (3). This adaptation aims to capture the influence of neuromodulators on neural excitability, namely dopamine via its D1 type receptor (D1R). In cell biology, D1R pathway activation increases the excitatory postsynaptic currents of AMPA and NMDA channels, as well as their synthesis and membrane expression (4). In the model, we assume that dopamine enhances the excitatory input current of a node.
The model with the implementation of dopamine neuromodulation is the following:

The variables correspond to the firing rate r, the mean membrane potential v, the adaptation u, the excitatory and inhibitory inputs sa and sg, the concentration of the neuromodulator Dp, and its receptor occupancy Md. Due to the relatively slow timescale of dopamine concentration changes, the variables Dp and Md have significantly higher time constants compared to the others.
Excitatory and inhibitory conductance-based synapses are included via AMPA receptors of maximal conductance ga and reversal potential Ea, and with GABA receptors of maximal conductance gg and reversal potential Eg, respectively. The excitatory synaptic input to a node, sa, depends on the signal received from all the other nodes connected to it, cexc, scaled by a factor sja. Moreover, it accounts for the node’s self-excitatory input r scaled by Ja. Similarly, the inhibitory input sg includes self-inhibition and external inputs from other connected nodes. 
The effect of the neuromodulator is embedded by the multiplicative term (Bd+Md) that affects the excitatory current of the node, where Bd is the baseline accounting for synapses in the absence of dopamine. Md is modeled with a classical sigmoidal dose-response curve dependent on the dopamine concentration Dp at the node, where Rd is its number of receptors for the neuromodulator (see also later) and sd and Z are parameters related to the receptor’s affinity and occupancy. The average concentration of dopamine Dp at the node depends on the dopaminergic inputs cdopa received by the node, scaled by a factor k, and on the reuptake of the neurotransmitter. This latter is described by the Michaelis-Menten equation for the Dopamine Transporter (DAT), which reuptakes the dopamine from the synaptic cleft at a maximal rate Vmax and with Michaelis constant Km.

**CONNECTIVITY IMPLEMENTATION**

The model allows for three different types of connections, glutamatergic, GABAergic, and dopaminergic. The inputs received by a node can change the activity of the node’s AMPA or GABA synapses or its dopamine concentration via the cexc, cinh, cdopa terms respectively. In general 

where 
●	p is a specific type of projection (glutamatergic, GABAergic, or dopaminergic), 
●	w is a scaling factor, kij is the weight of the connection between node i and node j taken from the matrix of the weights of the connectome obtained with tractography,
●	mp, ij is a value that can be either 0 or 1 depending on the type of projection between nodes i and j taken from the so-called “connectivity masks”,
●	r is the input firing rate received by node i from node j. 
The connectivity masks are boolean matrices specific for type of connection (glutamatergic, GABAergic, or dopaminergic). For a given connection type, mij = 1 if nodes i  and j are connected via that type; otherwise, mij = 0. Multiplying the tractography matrix by the corresponding mask isolates the inputs that a node receives from connections of a specific type, ensuring the node processes only inputs linked by the designated connectivity pattern. Two assumptions are made. First, while a node can send and receive connections of different types, any single connection to a node is of one type only. Second, because tractography imaging cannot determine the nature of a connection, the masks must be constructed using anatomical knowledge.
To address this, we created a set of connectivity masks based on established anatomical pathways ( 5, 6, 7, 8, 9, 10), including:
●	The classical GABAergic circuitry of the basal ganglia, excluding the subthalamic nucleus, and considering the Substantia Nigra pars reticulata (SNr) as part of the Globus pallidus (PA)
●	GABAergic projections from the Putamen (PU) and the Caudate nucleus (CA) to the Substantia Nigra pars compacta (SNc) and the PA and from the Amygdala and the Nucleus accumbens (AC) to the Ventral Tegmental Area (VTA).
●	Dopaminergic projections from the VTA to the amygdala, AC, prefrontal cortex, and hippocampus, and from the SN to the PU and CA.  
●	Glutamatergic projections from the prefrontal cortex to the VTA.
We provide the masks for three types of cortical parcellation and subcortical segmentation: the Desikan-Killiany atlas (70 cortical and 14 subcortical regions), the Schaefer 7 Networks atlas (100 cortical and 14 subcortical regions) and the AAL2 atlas (84 cortical and 36 subcortical regions). To these, 4 extra regions were added, namely the Left and Right Substantia nigra and Ventral Tegmental Area.
Since with the current imaging resolution it is difficult to get a precise measure of the weights of the dopaminergic nuclei connections, two strategies were adopted to adjust the connectome accordingly:
1.	For the DK atlas, weights of the connections to and from the SN and the VTA were included according to the values reported by Handfield-Jones et al (11). In this study, the authors used high-resolution MRI data from the human connectome project probabilistic tractography to measure SNc and VTA connectivity to the dorsal and ventral striatum and the prefrontal cortex. 
2.	For the Schaefer atlas and the AAL2, since region labelling does not follow canonical anatomical names, we used the D1 receptor density of each area as a surrogate for the connectivity weight between the specific area and the SN or the VTA.

Figure 2. In order, the excitatory, inhibitory and dopaminergic connectivity masks multiplied by the weights of the connectome in the Desikan-Killiany atlas. 

D1 receptors
The Rd parameters of the nodes receiving dopamine innervation is given by the normalized value of the number of receptors of the corresponding brain area. This information has been retrieved by realigning and resampling the MRI-PET coregistration of Kaller et al (12) to the parcellations of the two atlases. 

Figure 3. Dopamine D1R normalized values of the nodes receiving dopaminergic innervation in Desikan-Killiany atlas. 

REFERENCES
1.	Marmaduke Woodman, Abolfazl Ziaeemehr, vbjax, doi: 10.5281/zenodo.14204249.
2.	Gast R, Solla SA, Kennedy A. Neural heterogeneity controls computations in spiking neural networks. Proc Natl Acad Sci USA. 2024 Jan 16;121(3):e2311885121. doi: 10.1073/pnas.2311885121. Epub 2024 Jan 10. PMID: 38198531; PMCID: PMC10801870.
3.	Depannemaecker D., Duprat D., Angiolelli M., Sales Carbonell C., Wang H., Petkoski S., Sorrentino P., Sheheitli H., Jirsa V. (2024). A neural mass model with neuromodulation. bioRxiv, https://doi.org/10.1101/2024.06.23.600260
4.	Surmeier, D. J., Ding, J., Day, M., Wang, Z., & Shen, W. (2007). D1 and D2 dopamine-receptor modulation of striatal glutamatergic signaling in striatal medium spiny neurons. Trends in neurosciences, 30(5), 228–235. https://doi.org/10.1016/j.tins.2007.03.008
5.	Zikereya T, Shi K, Chen W. Goal-directed and habitual control: from circuits and functions to exercise-induced neuroplasticity targets for the treatment of Parkinson's disease. Front Neurol. 2023 Oct 10;14:1254447. doi: 10.3389/fneur.2023.1254447. PMID: 37881310; PMCID: PMC10597699.
6.	Mallet N, Delgado L, Chazalon M, Miguelez C, Baufreton J. Cellular and Synaptic Dysfunctions in Parkinson's Disease: Stepping out of the Striatum. Cells. 2019 Aug 29;8(9):1005. doi: 10.3390/cells8091005. PMID: 31470672; PMCID: PMC6769933.
7.	Russo SJ, Nestler EJ. The brain reward circuitry in mood disorders. Nat Rev Neurosci. 2013 Sep;14(9):609-25. doi: 10.1038/nrn3381. Epub 2013 Aug 14. Erratum in: Nat Rev Neurosci. 2013 Oct;14(10):736. PMID: 23942470; PMCID: PMC3867253.
8.	Zhang WH, Zhang JY, Holmes A, Pan BX. Amygdala Circuit Substrates for Stress Adaptation and Adversity. Biol Psychiatry. 2021 May 1;89(9):847-856. doi: 10.1016/j.biopsych.2020.12.026. Epub 2021 Jan 8. PMID: 33691931.
9.	Sonne J, Reddy V, Beato MR. Neuroanatomy, Substantia Nigra. [Updated 2024 Sep 10]. In: StatPearls [Internet]. Treasure Island (FL): StatPearls Publishing; 2025 Jan-. Available from: https://www.ncbi.nlm.nih.gov/books/NBK536995/
10.	Douma EH, de Kloet ER. Stress-induced plasticity and functioning of ventral tegmental dopamine neurons. Neurosci Biobehav Rev. 2020 Jan;108:48-77. doi: 10.1016/j.neubiorev.2019.10.015. Epub 2019 Oct 27. PMID: 31666179.
11.	Handfield-Jones, Nicholas, "Connectomic Analysis of Substantia Nigra Pars Compacta and Ventral Tegmental Area Projections to the Striatum and Cortex" (2019). Electronic Thesis and Dissertation Repository. 6463. https://ir.lib.uwo.ca/etd/6463.
12.	Kaller S, Rullmann M, Patt M, Becker GA, Luthardt J, Girbardt J, Meyer PM, Werner P, Barthel H, Bresch A, Fritz TH, Hesse S, Sabri O. Test-retest measurements of dopamine D1-type receptors using simultaneous PET/MRI imaging. Eur J Nucl Med Mol Imaging. 2017 Jun;44(6):1025-1032. doi: 10.1007/s00259-017-3645-0. Epub 2017 Feb 14. PMID: 28197685.
