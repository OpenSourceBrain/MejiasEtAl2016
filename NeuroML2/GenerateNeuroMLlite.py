from neuromllite import Network, Cell, InputSource, Population, Synapse,RectangularRegion,RandomLayout
from neuromllite import Projection, RandomConnectivity, Input, Simulation
from neuromllite.NetworkGenerator import generate_and_run
from neuromllite.NetworkGenerator import generate_neuroml2_from_network
import sys

################################################################################
###   Build new network

net = Network(id='MejiasFig2')
net.notes = 'Testing...'

net.parameters = { 'input':5 }

l23ecell = Cell(id='L23_E', lems_source_file='Prototypes.xml')
l23icell = Cell(id='L23_I', lems_source_file='RateBased.xml') #  hack to include this file too.  
l5ecell = Cell(id='L5_I', lems_source_file='NoisyCurrentSource.xml') #  hack to include this file too.  
l5icell = Cell(id='L5_I', lems_source_file='Prototypes.xml')


net.cells.append(l23ecell)
net.cells.append(l23icell)
net.cells.append(l5ecell)
net.cells.append(l5icell)


input_source0 = InputSource(id='iclamp0', 
                           pynn_input='DCSource', 
                           parameters={'amplitude':2, 'start':50., 'stop':150.})
                           
net.input_sources.append(input_source0)

r1 = RectangularRegion(id='network', x=0,y=0,z=0,width=100,height=100,depth=10)
net.regions.append(r1)

colors = [[8,48,107],         # dark-blue
          [228,26,28]]
    
color_str = {}
for i in range(len(colors)):
    color_str[i] = ''
    for c in colors[i]:
        color_str[i]+='%s '%(c/255.)
    color_str[i] = color_str[i][:-1]

pE = Population(id='L23_E', size=1, component=l23ecell.id, properties={'color':color_str[0]},random_layout = RandomLayout(region=r1.id))
pI = Population(id='L23_I', size=1, component=l23icell.id, properties={'color':color_str[1]},random_layout = RandomLayout(region=r1.id))

net.populations.append(pE)
net.populations.append(pI)

pops = [pE,pI]


net.synapses.append(Synapse(id='rs', 
                            lems_source_file='Prototypes.xml'))
                            

W = [[2.4167,   -0.3329],
    [2.9706,   -3.4554]]
W = [[0,   0],
    [2.9706,   0]]
    
for pre in pops:
    for post in pops:
        
        weight = W[pops.index(post)][pops.index(pre)]
        print('Connection %s -> %s weight %s'%(pre.id, post.id, weight))
        if weight!=0:
            
            net.projections.append(Projection(id='proj_%s_%s'%(pre.id,post.id),
                                              presynaptic=pre.id, 
                                              postsynaptic=post.id,
                                              synapse='rs',
                                              type='continuousProjection',
                                              delay=0,
                                              weight=weight,
                                              random_connectivity=RandomConnectivity(probability=1)))
                               
                        
net.inputs.append(Input(id='modulation',
                        input_source=input_source0.id,
                        population=pE.id,
                        percentage=100))

print(net)
print(net.to_json())
new_file = net.to_json_file('%s.json'%net.id)


################################################################################
###   Build Simulation object & save as JSON

sim = Simulation(id='Sim%s'%net.id,
                 network=new_file,
                 duration='200',
                 dt='0.025',
                 recordTraces={'all':'*'},
                 recordRates={'all':'*'})
                 
sim.to_json_file()



################################################################################
###   Run in some simulators

from neuromllite.NetworkGenerator import check_to_generate_or_run
import sys

check_to_generate_or_run(sys.argv, sim)

