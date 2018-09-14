from neuromllite import Network, Cell, InputSource, Population, Synapse,RectangularRegion,RandomLayout
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import sys
import numpy

     
def generate(wee = 1.5, wei = -3.25, wie = 3.5, wii = -2.5, sigma23=.3, sigma56=.45, duration=1000, dt = 0.025):

    ################################################################################
    ###   Build new network

    net = Network(id='MejiasFig2')
    net.notes = 'Testing...'

    net.parameters = { 'wee': wee,
                       'wei': wei,
                       'wie': wie,
                       'wii': wii,
                       'sigma23': sigma23,
                       'sigma56': sigma56 }

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


    W = [['wee',   'wei'],
        ['wie',   'wii']]

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
                     duration=duration,
                     dt=dt,
                     recordRates={'all':'*'})

    sim.to_json_file()

    return sim, net
                                        
if __name__ == "__main__":
    
    from neuromllite.NetworkGenerator import check_to_generate_or_run
    from neuromllite.sweep.ParameterSweep import *
    from pyneuroml import pynml
    
    import sys
    
    if '-sweep' in sys.argv:
        pass
    
    if '-test' in sys.argv:
        
        pop_colors = {'L23_E':'#dd7777','L23_I':'#7777dd','L23_E Py':'#990000','L23_I Py':'#000099',
                      'L56_E':'#77dd77','L56_I':'#dd77dd','L56_E Py':'#009900','L56_I Py':'#990099'}
        
        sim, net = generate(wee = 0, wei = 0, wie = 0, wii = 0, duration=50000, dt=0.2)
        sim, net = generate(duration=50000, dt=0.2)
        
        simulator = 'jNeuroML'

        nmllr = NeuroMLliteRunner('%s.json'%sim.id,
                                  simulator=simulator)
                       
        incl_23 = False               
        incl_23 = True    
        incl_56 = False
        incl_56 = True
        traces, events = nmllr.run_once('/tmp')
        xs = []
        ys = []
        labels = []
        colors = []
        histxs = []
        histys = []
        histlabels = []
        histcolors = []
        bins = 150
        
        for tr in traces:
            if tr!='t':
                if ('23' in tr and incl_23) or ('56' in tr and incl_56):
                    xs.append(traces['t'])
                    ys.append(traces[tr])
                    pop = tr.split('/')[0]
                    labels.append(pop)
                    colors.append(pop_colors[pop])

                    hist1, edges1 = numpy.histogram(traces[tr],bins=bins)
                    mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histcolors.append(pop_colors[pop])
                    histlabels.append(pop)

        
        debug_datafile = '../Python/debug/intralaminar/simulation_Iext_0_nrun_0.txt'
        
        with open(debug_datafile) as f:
            l23e = []; l23i = []; l56e = []; l56i = []; ts = []
            t=0
            dt = 0.0002
            count = 0
            for line in f:
                w = line.split()
                l23e.append(float(w[0]))
                l23i.append(float(w[1]))
                l56e.append(float(w[2]))
                l56i.append(float(w[3]))
                ts.append(t)
                t+=dt
                count+=1
            print("Read in 4 x %i data points from %s"%(count, debug_datafile))
                
            
            if incl_23:
                xs.append(ts)
                ys.append(l23e)
                pop = 'L23_E Py'
                labels.append(pop)
                colors.append(pop_colors[pop])
                
                hist1, edges1 = numpy.histogram(l23e,bins=bins)
                mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                histxs.append(mid1)
                histys.append(hist1)
                histlabels.append(pop)
                histcolors.append(pop_colors[pop])
                
            
                xs.append(ts)
                ys.append(l23i)
                pop = 'L23_I Py'
                labels.append(pop)
                colors.append(pop_colors[pop])
                
                hist1, edges1 = numpy.histogram(l23i,bins=bins)
                mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                histxs.append(mid1)
                histys.append(hist1)
                histlabels.append(pop)
                histcolors.append(pop_colors[pop])
                
            if incl_56:
                xs.append(ts)
                ys.append(l56e)
                pop = 'L56_E Py'
                labels.append(pop)
                colors.append(pop_colors[pop])
                
                hist1, edges1 = numpy.histogram(l56e,bins=bins)
                mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                histxs.append(mid1)
                histys.append(hist1)
                histlabels.append(pop)
                histcolors.append(pop_colors[pop])
                
                xs.append(ts)
                ys.append(l56i)
                pop = 'L56_I Py'
                labels.append(pop)
                colors.append(pop_colors[pop])
                
                hist1, edges1 = numpy.histogram(l56i,bins=bins)
                mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                histxs.append(mid1)
                histys.append(hist1)
                histlabels.append(pop)
                histcolors.append(pop_colors[pop])
            
        print colors
        pynml.generate_plot(xs,
                            ys,
                            'Comparison with connections',
                            labels=labels, 
                            colors=colors, 
                            show_plot_already=False)
                            
        pynml.generate_plot(histxs,
                            histys,
                            'Histograms',
                            labels=histlabels, 
                            colors=histcolors, 
                            show_plot_already=False, 
                            markers=['o' for x in histxs], 
                            markersizes=[2 for x in histxs])
        
        import matplotlib.pyplot as plt

        plt.show()
        

    else:
        
        sim, net = generate()
        
        ################################################################################
        ###   Run in some simulators


        check_to_generate_or_run(sys.argv, sim)

