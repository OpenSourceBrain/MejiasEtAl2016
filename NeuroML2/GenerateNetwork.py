from neuromllite import *

import random

colors = {}
centres = {}

scale = 10
f = open('MERetal14_on_F99.tsv')
for l in f:
    w = l.split()
    id = w[0].replace('-','_')
    colors[id] = w[1]
    centres[id] = (float(w[2])*scale,float(w[3])*scale,float(w[4])*scale)

print centres
    

################################################################################
###   Build a new network

net = Network(id='MejiasNetwork')
net.notes = "...."
net.parameters = {}

l23ecell = Cell(id='L23_E', lems_source_file='Prototypes.xml')
l23icell = Cell(id='L23_I', lems_source_file='RateBased.xml') #  hack to include this file too.  

net.cells.append(l23ecell)
net.cells.append(l23icell)


net.synapses.append(Synapse(id='rs', 
                            lems_source_file='NoisyCurrentSource.xml')) #  hack to include this file too.  
                            
net.parameters['stim_amp'] = '1nA'

input_source_0 = InputSource(id='iclamp_0', 
                           neuroml2_input='PulseGenerator', 
                           parameters={'amplitude':'stim_amp', 'delay':'50ms', 'duration':'400ms'})
                           
input_source_1 = InputSource(id='iclamp_1', 
                           neuroml2_input='PulseGenerator', 
                           parameters={'amplitude':'stim_amp', 'delay':'550ms', 'duration':'400ms'})

net.input_sources.append(input_source_0)
net.input_sources.append(input_source_1)



f = open('Neuron_2015_Table.csv')
all_tgts = []

for l in f:
    #print l
    w = l.split(',')
    tgt = w[0]
    src = w[1]
    if tgt!='TARGET':
        if not tgt in all_tgts: #and 'V' in tgt and 'V' in src:
            all_tgts.append(tgt)
        
print all_tgts
print(len(all_tgts))

f = open('Neuron_2015_Table.csv')
pop_ids = []
used_ids = {}


for l in f:
    #print l
    w = l.split(',')
    tgt = w[0]
    src = w[1]
    if tgt!='TARGET' and src in all_tgts:
        fln = float(w[2])
        print('Adding conn from %s to %s with FLN: %s'%(src,tgt,fln))
        for pop_id in [tgt,src]:
            if not pop_id in pop_ids:
                
                repl = pop_id.replace('/','_')
                
                p = centres[repl]
                used_ids[pop_id] = '_%s'%repl if repl[0].isdigit() else repl
                
                r = RectangularRegion(id='Region_%s'%used_ids[pop_id], x=p[0],y=p[1],z=p[2],width=1,height=1,depth=1)
                net.regions.append(r)

                p0 = Population(id=used_ids[pop_id], 
                                size=1, 
                                component=l23ecell.id, 
                                properties={'color':'%s %s %s'%(random.random(),random.random(),random.random()),
                                            'radius':scale},
                                random_layout = RandomLayout(region=r.id))

                net.populations.append(p0)
                pop_ids.append(pop_id)


        ################################################################################
        ###   Add a projection

        if fln>0:
            net.projections.append(Projection(id='proj_%s_%s'%(used_ids[src],used_ids[tgt]),
                                              presynaptic=used_ids[src], 
                                              postsynaptic=used_ids[tgt],
                                              synapse='rs',
                                              type='continuousProjection',
                                              weight=fln,
                                              random_connectivity=RandomConnectivity(probability=1)))



stim_pops = ['V1', '_8m']
stim_ids = [input_source_0.id, input_source_1.id]

for i in range(len(stim_pops)):
    pop = stim_pops[i]
    stim = stim_ids[i]
    net.inputs.append(Input(id='Stim_%s'%pop,
                            input_source=stim,
                            population=pop,
                            percentage=100))

print(net)

print(net.to_json())
new_file = net.to_json_file('%s.json'%net.id)

sim = Simulation(id='Sim%s'%net.id,
                 network=new_file,
                 duration='1000',
                 dt='0.025',
                 recordRates={'all':'*'})


################################################################################
###   Export to some formats
###   Try:
###        python Example1.py -graph2

from neuromllite.NetworkGenerator import check_to_generate_or_run
from neuromllite import Simulation
import sys

check_to_generate_or_run(sys.argv, sim)

