## Mejias et al. 2016

Implementation in Python and in NeuroML2/LEMS of Jorge F. Mejias, John D. Murray, Henry Kennedy, and Xiao-Jing Wang, “Feedforward and Feedback Frequency-Dependent Interactions in a Large-Scale Laminar Network of the Primate Cortex.” [Science Advances](http://advances.sciencemag.org/content/2/11/e1601335), 2016 ([bioRxiv](https://doi.org/10.1101/065854)).

### The model
The model simulates the dynamics of a cortical laminar structure across multiple scales: (I) intralaminar, (II) interlaminar, (III) interareal, (IV) whole cortex. Interestingly, the authors show that while feedforward pathways are associated with gamma oscillations (30 - 70 Hz), feedback pathways are modulated by alpha/low beta oscillations (8 - 15 Hz).

<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/Mejias-2016.png" width="500px"/>
<sup><i>Interareal model (work in progress)</i></sup>
<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/interareal.png" width="500px"/>
<sup><i>View of whole cortex model on Open Source Brain (work in progress)</i></sup>
<img src="https://raw.githubusercontent.com/OpenSourceBrain/MejiasEtAl2016/master/NeuroML2/img/OSB1.png" width="500px"/>

> Note: This repo is a work in progress. So far this repository contains the implementation for the model dynamics at the intralaminar and the interlaminar level. At the moment we are working on the implementation of the interareal level.

### Simulation of the model

#### Python
So far, we have reproduced the main findings described by Mejias et al., 2016 at the [intralaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/intralaminar.py) and [interlaminar](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/Python/interlaminar.py) level. The main results are described [here](Python/README.md).

#### NeuroML2
A basic implementation and simulation of the intralaminar model have also been implemented in NeuroML2/LEMS. [GenerateNeuroMLlite.py](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/NeuroML2/GenerateNeuroMLlite.py) generates the LEMS file with the description of the network.

The simulation can be run by calling inside the NeuroML2 folder:

    python GenerateNeuroMLlite.py -jnml

### Requirements
The necessary Python packages are listed on the [requirements.txt](https://github.com/OpenSourceBrain/MejiasEtAl2016/blob/master/requirements.txt) file.
