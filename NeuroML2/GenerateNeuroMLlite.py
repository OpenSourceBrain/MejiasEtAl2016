from neuromllite import Network, Cell, InputSource, Population, Synapse,RectangularRegion,RandomLayout
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import sys
import numpy


def generate(wee = 1.5, wei = -3.25, wie = 3.5, wii = -2.5, interlaminar1=0,
             interlaminar2=0, sigma23=.3, sigma56=.45, noise=True, duration=1000, dt=0.2, Iext=0):

    ################################################################################
    ###   Build new network

    net = Network(id='MejiasFig2')
    net.notes = 'Testing...'
    if dt!=0.2 and dt!=0.02:
        print('Using a value for dt which is not supported!!')
        quit()

    net.parameters = { 'wee': wee,
                       'wei': wei,
                       'wie': wie,
                       'wii': wii,
                       'interlaminar1': interlaminar1,
                       'interlaminar2': interlaminar2,
                       'sigma23': sigma23,
                       'sigma56': sigma56 }

    suffix = '' if noise else '_flat'
    suffix2 = '' if dt == 0.2 else '_smalldt'
    l23ecell = Cell(id='L23_E'+suffix+suffix2, lems_source_file='Prototypes.xml')
    l23icell = Cell(id='L23_I'+suffix+suffix2, lems_source_file='RateBased.xml') #  hack to include this file too.
    l56ecell = Cell(id='L56_E'+suffix+suffix2, lems_source_file='NoisyCurrentSource.xml') #  hack to include this file too.
    l56icell = Cell(id='L56_I'+suffix+suffix2, lems_source_file='Prototypes.xml')


    net.cells.append(l23ecell)
    net.cells.append(l23icell)
    net.cells.append(l56ecell)
    net.cells.append(l56icell)


    input_source0 = InputSource(id='iclamp0',
                               pynn_input='DCSource',
                               parameters={'amplitude':Iext, 'start':0, 'stop':duration})

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


    def internal_connections(pops):
        for pre in pops:
            for post in pops:

                weight = W[pops.index(pre)][pops.index(post)]
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

    l2e_l2e = 'wee'; l2e_l2i = 'wei'; l2i_l2e = 'wie'; l2i_l2i = 'wii';
    l5e_l5e = 'wee'; l5e_l5i = 'wei'; l5i_l5e = 'wie'; l5i_l5i = 'wii';
    l2e_l5i = 0; l2e_l5e = 'interlaminar2'; l2i_l5e = 0; l2i_l5i = 0;
    l5e_l2e = 0; l5e_l2i= 'interlaminar1'; l5i_l2e = 0; l5i_l2i = 0;
    W = [[l2e_l2e, l2e_l2i, l2e_l5e, l2e_l5i],
         [l2i_l2e, l2i_l2i, l2i_l5e, l2i_l5i],
         [l5e_l2e, l5e_l2i, l5e_l5e, l5e_l5i],
         [l5i_l2e, l5i_l2i, l5i_l5e, l5i_l5i]]
    pops = [pl23e,pl23i, pl56e, pl56i]
    internal_connections(pops)

    # Add modulation
    net.inputs.append(Input(id='modulation_l23_E',
                            input_source=input_source0.id,
                            population=pl23e.id,
                            percentage=100))
    net.inputs.append(Input(id='modulation_l56_E',
                            input_source=input_source0.id,
                            population=pl56e.id,
                            percentage=100))


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
    JEE = 1.5
    JIE = 3.5
    JEI = -3.25
    JII = -2.5

    pop_colors = {'L23_E':'#dd7777','L23_I':'#7777dd','L23_E Py':'#990000','L23_I Py':'#000099',
                  'L56_E':'#77dd77','L56_I':'#dd77dd','L56_E Py':'#009900','L56_I Py':'#990099'}

    if '-sweep' in sys.argv:
        pass

    if '-test' in sys.argv or '-dt' in sys.argv:

        if '-test' in sys.argv:

            arg_options = {'No connections; no noise':[{'wee':0, 'wei':0, 'wie':0, 'wii':0,
                                                        'duration':1000, 'dt':0.2, 'noise':False},
                                                        'simulation_Iext0_nrun0_noise0.0_dur1.0_noconns_dt0.0002.txt'],
                           'With connections; no noise':[{'wee':JEE, 'wei':JIE, 'wie':JEI, 'wii':JII,
                                                          'duration':1000, 'dt':0.2, 'noise':False},
                                                          'simulation_Iext0_nrun0_noise0.0_dur1.0_dt0.0002.txt'],
                            'No connections; with noise':[{'wee':0, 'wei':0, 'wie':0, 'wii':0,
                                                        'duration':50000, 'dt':0.2, 'noise':True},
                                                        'simulation_Iext0_nrun0_noiseNone_dur50.0_noconns_dt0.0002.txt'],
                           'With connections; with noise':[{'wee':JEE, 'wei':JIE, 'wie':JEI, 'wii':JII,
                                                          'duration':50000, 'dt':0.2, 'noise':True},
                                                          'simulation_Iext0_nrun0_noiseNone_dur50.0_dt0.0002.txt']}
                                                          

            hist_bins = 50

        elif '-dt' in sys.argv:
            print('Running dt tests...')

            arg_options = {'dt normal':[{'wee':0, 'wei':0, 'wie':0, 'wii':0,
                                                        'duration':50000, 'dt':0.2, 'noise':True},
                                                        'simulation_Iext0_nrun0_noiseNone_dur50.0_noconns_dt0.0002.txt'],
                           'dt small':[{'wee':0, 'wei':0, 'wie':0, 'wii':0,
                                                        'duration':50000, 'dt':0.02, 'noise':True},
                                                        'simulation_Iext0_nrun0_noiseNone_dur50.0_noconns_dt2e-05.txt']}

            hist_bins = 50

        #sim, net = generate(wee = 0, wei = 0, wie = 0, wii = 0, duration=50000, dt=0.2)
        #sim, net = generate(duration=50000, dt=0.2)

        for a in arg_options:
            print("Running sim: %s"%arg_options[a])

            sim, net = generate(**arg_options[a][0])

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

            for tr in traces:
                if tr!='t':
                    if ('23' in tr and incl_23) or ('56' in tr and incl_56):
                        xs.append(traces['t'])
                        ys.append(traces[tr])
                        pop = tr.split('/')[0]
                        labels.append(pop)
                        colors.append(pop_colors[pop])

                        hist1, edges1 = numpy.histogram(traces[tr],bins=hist_bins)
                        mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                        histxs.append(mid1)
                        histys.append(hist1)
                        histcolors.append(pop_colors[pop])
                        histlabels.append(pop)


            debug_datafile = '../Python/debug/intralaminar/%s'%arg_options[a][1]

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

                    hist1, edges1 = numpy.histogram(l23e,bins=hist_bins)
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

                    hist1, edges1 = numpy.histogram(l23i,bins=hist_bins)
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

                    hist1, edges1 = numpy.histogram(l56e,bins=hist_bins)
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

                    hist1, edges1 = numpy.histogram(l56i,bins=hist_bins)
                    mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histlabels.append(pop)
                    histcolors.append(pop_colors[pop])

            print colors
            pynml.generate_plot(xs,
                                ys,
                                a,
                                labels=labels,
                                linewidths=[(1 if 'Py' in l else 2) for l in labels],
                                colors=colors,
                                show_plot_already=False,
                                yaxis='Rate (Hz)',
                                xaxis='Time (s)',
                                legend_position='right',
                                title_above_plot=True)

            if arg_options[a][0]['noise']:
                pynml.generate_plot(histxs,
                                    histys,
                                    'Histograms: %s'%a,
                                    labels=histlabels,
                                    colors=histcolors,
                                    show_plot_already=False,
                                    xaxis='Rate bins (Hz)',
                                    yaxis='Num timesteps rate in bins',
                                    markers=['o' for x in histxs],
                                    markersizes=[2 for x in histxs],
                                    legend_position='right',
                                    title_above_plot=True)

            import matplotlib.pyplot as plt

        plt.show()

    elif '-intralaminar' in sys.argv:
        import numpy as np
        sys.path.append("../Python")

        from intralaminar import intralaminar_analysis, intralaminar_plt

        wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = 0; l2e_l5e = 0
        # Input strength of the excitatory population
        Iexts = [0, 2, 4, 6]
        simulation = {}

        for Iext in Iexts:
            sim, net = generate(wee=wee, wei=wei, wie=wie, wii=wii, interlaminar1=l5e_l2i, interlaminar2=l2e_l5e, duration=25000,
                                Iext=Iext)
            ################################################################################
            ###   Run in some simulators

            check_to_generate_or_run(sys.argv, sim)
            simulator = 'jNeuroML'

            nmllr = NeuroMLliteRunner('%s.json'%sim.id,
                              simulator=simulator)

            # total number of simulations to run for each input strength
            nruns = 10
            simulation[Iext] = {}
            for run in range(nruns):
                simulation[Iext][run] = {}
                traces, events = nmllr.run_once('/tmp')
                # For the purpose of this analysis we will save only the traces related to the excitatory L23 population
                simulation[Iext][run]['L23_E/0/L23_E/r'] = np.array(traces['L23_E/0/L23_E/r'])

        # analyse the traces using python methods

        # transform the results into a numpy array and analyse them
        psd_dic = intralaminar_analysis(simulation, Iexts, nruns, layer='L23', dt=2e-04, transient=5)
        # plot the results
        intralaminar_plt(psd_dic)


    elif '-interlaminar' in sys.argv:

        wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = .75; l2e_l5e = 1
        sim, net = generate(wee=wee, wei=wei, wie=wie, wii=wii, interlaminar1=l5e_l2i, interlaminar2=l2e_l5e, Iext=2)
        ################################################################################
        ###   Run in some simulators

        check_to_generate_or_run(sys.argv, sim)

    else:

        sim, net = generate()

        ################################################################################
        ###   Run in some simulators


        check_to_generate_or_run(sys.argv, sim)

