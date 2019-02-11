## Mejias-2016
Implementation in Matlab, Python and of NeuroML2/LEMS and Mejias, Jorge F., John D. Murray, Henry Kennedy, and Xiao-Jing Wang, “Feedforward and Feedback Frequency-Dependent Interactions in a Large-Scale Laminar Network of the Primate Cortex.” [Science Advances](http://advances.sciencemag.org/content/2/11/e1601335), 2016.

### The model
The model simulates the dynamics of a cortical laminar structure across multiple scales: (I) intralaminar, (II) interlaminar, (III) interareal, (IV) whole cortex. Interestingly, the authors show that while feedforward pathways are associated with gamma oscillations (30 - 70 Hz), feedback pathways are modulated by alpha/low beta oscillations (8 - 15 Hz).

<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/Mejias-2016.png" width="500px"/>

> Note: This repo is a work in progress. So far this repository contains the implementation for the model dynamics at the intralaminar and the interlaminar level. At the moment I am working on the implementation of the interareal level.

### The Simulation

#### Matlab
This folder contains the original model developed by Jorge Mejias and available on [ModelDb](https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=249589&file=/Mejias2016/readme.html#tabs-2).

#### Python
So far, we have reproduced the main findings described by Mejias et al., 2016 at the [intralaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/intralaminar.py) and [interlaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/interlaminar.py) level. The main results are described [here](Python/README.md).

#### NeuroML2
A basic implementation and simulation of the intralaminar model have also been implemented in NeuroML2/LEMS. [GenerateNetwork.py](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/NeuroML2/GenerateNetwork.py) generates the LEMS file with the description of the network.

The simulation can be run by calling inside the NeuroML2 folder:

    python GenerateNetwork.py -jnml

### Requirements
The necessary Python packages are listed on the [requirements.txt](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/requirements.txt) file.
