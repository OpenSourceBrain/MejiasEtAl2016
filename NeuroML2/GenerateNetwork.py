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

net = Network(id='net0')
net.notes = "...."
net.parameters = {}

cell = Cell(id='testcell', pynn_cell='IF_cond_alpha')
cell.parameters = { "tau_refrac":5, "i_offset":0 }
net.cells.append(cell)

net.synapses.append(Synapse(id='ampa', 
                            pynn_receptor_type='excitatory', 
                            pynn_synapse_type='cond_alpha', 
                            parameters={'e_rev':0, 'tau_syn':2}))
                            
net.parameters['stim_amp'] = '850pA'
input_source = InputSource(id='iclamp_0', 
                           neuroml2_input='PulseGenerator', 
                           parameters={'amplitude':'stim_amp', 'delay':'100ms', 'duration':'500ms'})

net.input_sources.append(input_source)



f = open('Neuron_2015_Table.csv')
all_tgts = []

for l in f:
    #print l
    w = l.split(',')
    tgt = w[0]
    src = w[1]
    if tgt!='TARGET':
        if not tgt in all_tgts:
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
                                component=cell.id, 
                                properties={'color':'%s %s %s'%(random.random(),random.random(),random.random()),
                                            'radius':scale},
                                random_layout = RandomLayout(region=r.id))

                net.populations.append(p0)
                pop_ids.append(pop_id)


        ################################################################################
        ###   Add a projection

        if fln>0.0:
            net.projections.append(Projection(id='proj_%s_%s'%(used_ids[src],used_ids[tgt]),
                                              presynaptic=used_ids[src], 
                                              postsynaptic=used_ids[tgt],
                                              synapse='ampa',
                                              weight=fln,
                                              random_connectivity=RandomConnectivity(probability=1)))



stim_pops = ['V1', '_8m']

for pop in stim_pops:
    net.inputs.append(Input(id='Stim_%s'%pop,
                            input_source=input_source.id,
                            population=pop,
                            percentage=100))

print(net)
net.id = 'TestNetwork'

print(net.to_json())
new_file = net.to_json_file('Example1_%s.json'%net.id)

sim = Simulation(id='SimExample',
                 network=new_file,
                 duration='700',
                 dt='0.025',
                 recordTraces={'all':'*'})


################################################################################
###   Export to some formats
###   Try:
###        python Example1.py -graph2

from neuromllite.NetworkGenerator import check_to_generate_or_run
from neuromllite import Simulation
import sys

check_to_generate_or_run(sys.argv, sim)

