from neuromllite import Network, Cell, InputSource, Population, Synapse,RectangularRegion,RandomLayout
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import sys

################################################################################
###   Build new network

net = Network(id='MejiasFig2')
net.notes = 'Testing...'

#net.parameters = { }

l23ecell = Cell(id='L23_E', lems_source_file='Prototypes.xml')
l23icell = Cell(id='L23_I', lems_source_file='RateBased.xml') #  hack to include this file too.  
l56ecell = Cell(id='L56_E', lems_source_file='NoisyCurrentSource.xml') #  hack to include this file too.  
l56icell = Cell(id='L56_I', lems_source_file='Prototypes.xml')


net.cells.append(l23ecell)
net.cells.append(l23icell)
net.cells.append(l56ecell)
net.cells.append(l56icell)


input_source0 = InputSource(id='iclamp0', 
                           pynn_input='DCSource', 
                           parameters={'amplitude':2, 'start':50., 'stop':150.})
                           
net.input_sources.append(input_source0)

l23 = RectangularRegion(id='L23', x=0,y=100,z=0,width=100,height=100,depth=10)
net.regions.append(l23)

l56 = RectangularRegion(id='L56', x=0,y=0,z=0,width=100,height=100,depth=10)
net.regions.append(l56)
    
color_str = {'l23e':'.8 0 0','l23i':'0 0 .8',
             'l56e':'1 .2 0','l56i':'0 .2 1'}

pl23e = Population(id='L23_E', size=1, component=l23ecell.id, properties={'color':color_str['l23e']},random_layout = RandomLayout(region=l23.id))
pl23i = Population(id='L23_I', size=1, component=l23icell.id, properties={'color':color_str['l23i']},random_layout = RandomLayout(region=l23.id))

pl56e = Population(id='L56_E', size=1, component=l56ecell.id, properties={'color':color_str['l56e']},random_layout = RandomLayout(region=l56.id))
pl56i = Population(id='L56_I', size=1, component=l56icell.id, properties={'color':color_str['l56i']},random_layout = RandomLayout(region=l56.id))

net.populations.append(pl23e)
net.populations.append(pl23i)

net.populations.append(pl56e)
net.populations.append(pl56i)



net.synapses.append(Synapse(id='rs', 
                            lems_source_file='Prototypes.xml'))
                            

wee = 1.5; wei = -3.25
wie = 3.5; wii = -2.5
W = [[wee,   wei],
    [wie,   wii]]
    
def internal_connections(pops):
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

pops = [pl23e,pl23i]
internal_connections(pops)

pops = [pl56e,pl56i]
internal_connections(pops)
                               
'''
net.inputs.append(Input(id='modulation',
                        input_source=input_source0.id,
                        population=pE.id,
                        percentage=100))'''

print(net)
print(net.to_json())
new_file = net.to_json_file('%s.json'%net.id)


################################################################################
###   Build Simulation object & save as JSON

sim = Simulation(id='Sim%s'%net.id,
                 network=new_file,
                 duration='1000',
                 dt='0.025',
                 recordRates={'all':'*'})
                 
sim.to_json_file()



################################################################################
###   Run in some simulators

from neuromllite.NetworkGenerator import check_to_generate_or_run
import sys

check_to_generate_or_run(sys.argv, sim)

