## Mejias et al. 2016

[![Continuous build using OMV](https://github.com/OpenSourceBrain/MejiasEtAl2016/actions/workflows/omv-ci.yml/badge.svg)](https://github.com/OpenSourceBrain/MejiasEtAl2016/actions/workflows/omv-ci.yml)

Implementation in in Matlab, Python and in NeuroML2/LEMS of Jorge F. Mejias, John D. Murray, Henry Kennedy, and Xiao-Jing Wang, “Feedforward and Feedback Frequency-Dependent Interactions in a Large-Scale Laminar Network of the Primate Cortex.” [Science Advances](http://advances.sciencemag.org/content/2/11/e1601335), 2016 ([bioRxiv](https://doi.org/10.1101/065854)).

### The model
The model simulates the dynamics of a cortical laminar structure across multiple scales: (I) intralaminar, (II) interlaminar, (III) interareal, (IV) whole cortex. Interestingly, the authors show that while feedforward pathways are associated with gamma oscillations (30 - 70 Hz), feedback pathways are modulated by alpha/low beta oscillations (8 - 15 Hz).

<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/Mejias-2016.png" width="500px"/>
<sup><i>Interareal model (work in progress)</i></sup>
<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/interareal.png" width="500px"/>
<sup><i>View of whole cortex model on Open Source Brain (work in progress)</i></sup>
<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/OSB1.png" width="500px"/>

> Note: This repo is a work in progress. So far this repository contains the implementation for the model dynamics at the intralaminar and the interlaminar level. At the moment we are working on the implementation of the interareal level.

### Simulation of the model

#### Matlab
This folder contains the original model developed by Jorge Mejias and available on [ModelDB](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=249589&file=/Mejias2016/readme.html#tabs-2).

#### Python
So far, we have reproduced the main findings described by Mejias et al., 2016 at the [intralaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/intralaminar.py) and [interlaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/interlaminar.py) level. The main results are described [here](Python/README.md).

#### NeuroML2
A basic implementation and simulation of the intralaminar model have also been implemented in NeuroML2/LEMS. [GenerateNeuroMLlite.py](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/NeuroML2/GenerateNeuroMLlite.py) generates the LEMS file with the description of the network.

The simulation can be run by calling inside the NeuroML2 folder:

    python GenerateNeuroMLlite.py -jnml


### Requirements
The necessary Python packages are listed on the [requirements.txt](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/requirements.txt) file.


### Data & code sources

As outlined above, the original Matlab scripts related to [Mejias et al. 2016](http://advances.sciencemag.org/content/2/11/e1601335) were taken from [ModelDB](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=249589&file=/Mejias2016/readme.html#tabs-2).

The data on 3D positions of areas ([MERetal14_on_F99.tsv](NeuroML2/MERetal14_on_F99.tsv)) was taken from https://scalablebrainatlas.incf.org/macaque/MERetal14#downloads; https://scalablebrainatlas.incf.org/services/regioncenters.php

- Markov NT, Ercsey-Ravasz MM, Ribeiro Gomes AR, Lamy C, Magrou L, Vezoli J, Misery P, Falchier A, Quilodran R, Gariel MA, Sallet J, Gamanut R, Huissoud C, Clavagnier S, Giroud P, Sappey-Marinier D, Barone P, Dehay C, Toroczkai Z, Knoblauch K, Van Essen DC, Kennedy H (2014) "A weighted and directed interareal connectivity matrix for macaque cerebral cortex." [Cereb Cortex 24(1):17-36.](http://dx.doi.org/10.1093/cercor/bhs270)

- Rembrandt Bakker, Paul Tiesinga, Rolf Kötter (2015) "The Scalable Brain Atlas: instant web-based access to public brain atlases and related content." Neuroinformatics. http://link.springer.com/content/pdf/10.1007/s12021-014-9258-x ([arXiv](http://arxiv.org/abs/1312.6310)) 


