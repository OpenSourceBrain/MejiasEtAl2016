from neuromllite import Network, Cell, InputSource, Population, Synapse,RectangularRegion,RandomLayout
from neuromllite import Projection, RandomConnectivity, Input, Simulation
import numpy as np
import pickle
import pandas as pd
from itertools import permutations

# Add the Python folder to the Python path
import sys
sys.path.append("../Python")

def generate(wee = 1.5, wei = -3.25, wie = 3.5, wii = -2.5,
             i_l5e_l2i=0., i_l2e_l5e=0.,
             areas=['V1'], FF_l2e_l2e=0., FB_l5e_l2i=0., FB_l5e_l5e=0., FB_l5e_l5i=0., FB_l5e_l2e=0.,
             sigma23=.3, sigma56=.45, noise=True, duration=1000, dt=0.2, Iext=[[0, 0]], count=0,
             net_id='MejiasFig2'):

    ################################################################################
    ###   Build new network

    net = Network(id=net_id)
    net.notes = 'Testing...'

    net.parameters = { 'wee': wee,
                       'wei': wei,
                       'wie': wie,
                       'wii': wii,
                       'l5e_l2i': i_l5e_l2i,
                       'l2e_l5e': i_l2e_l5e,
                       'FF_l2e_l2e': FF_l2e_l2e,
                       'FB_l5e_l2i': FB_l5e_l2i,
                       'FB_l5e_l5e': FB_l5e_l5e,
                       'FB_l5e_l5i': FB_l5e_l5i,
                       'FB_l5e_l2e': FB_l5e_l2e,
                       'sigma23': sigma23,
                       'sigma56': sigma56 }

    suffix = '' if noise else '_flat'
    
    if dt==0.2:
        suffix2 = ''
    elif dt==0.02:
        suffix2 = '_smalldt'
    else:
        print('Using a value for dt which is not supported!!')
        quit()
        
    l23ecell = Cell(id='L23_E_comp'+suffix+suffix2, lems_source_file='Prototypes.xml')
    l23icell = Cell(id='L23_I_comp'+suffix+suffix2, lems_source_file='RateBased.xml') #  hack to include this file too.
    l56ecell = Cell(id='L56_E_comp'+suffix+suffix2, lems_source_file='NoisyCurrentSource.xml') #  hack to include this file too.
    l56icell = Cell(id='L56_I_comp'+suffix+suffix2, lems_source_file='Prototypes.xml')


    net.cells.append(l23ecell)
    net.cells.append(l23icell)
    net.cells.append(l56ecell)
    net.cells.append(l56icell)


    color_str = {'l23e':'.8 0 0','l23i':'0 0 .8',
                 'l56e':'1 .2 0','l56i':'0 .2 1'}

    def internal_connections(pops, W, pre_pop, post_pop):
        print('Connection %s -> %s:' %(pre_pop.id[:2], post_pop.id[:2]))
        weight = str(W[pops.index(pre_pop)][pops.index(post_pop)])
        print('    Connection %s -> %s weight %s'%(pre_pop.id, post_pop.id, weight))
        if weight!=0:
            net.projections.append(Projection(id='proj_%s_%s'%(pre_pop.id, post_pop.id),
                                              presynaptic=pre_pop.id,
                                              postsynaptic=post_pop.id,
                                              synapse='rs',
                                              type='continuousProjection',
                                              delay=0,
                                              weight=weight,
                                              random_connectivity=RandomConnectivity(probability=1)))


    # Basic case when there is just one area
    l2e_l2e = 'wee'; l2e_l2i = 'wei'; l2i_l2e = 'wie'; l2i_l2i = 'wii';
    l5e_l5e = 'wee'; l5e_l5i = 'wei'; l5i_l5e = 'wie'; l5i_l5i = 'wii';
    l2e_l5i = 0; l2e_l5e = 'l2e_l5e'; l2i_l5e = 0; l2i_l5i = 0;
    l5e_l2e = 0; l5e_l2i= 'l5e_l2i'; l5i_l2e = 0; l5i_l2i = 0;
    W_tmp = np.array([[l2e_l2e, l2e_l2i, l2e_l5e, l2e_l5i],
                     [l2i_l2e, l2i_l2i, l2i_l5e, l2i_l5i],
                     [l5e_l2e, l5e_l2i, l5e_l5e, l5e_l5i],
                     [l5i_l2e, l5i_l2i, l5i_l5e, l5i_l5i]], dtype='U14')
    # Todo: change the areas[0]
    n_areas = len(areas[0])
    # Create the correct connectivity size matrix
    W = np.zeros((4 * n_areas, 4 * n_areas), dtype='U14')

    for i in range(n_areas):
        for j in range(n_areas):
            W[4*i:4*i+4, 4*j:4*j+4] = W_tmp

    # Define FF and FB connections
    W[0, 4] = 'FF_l2e_l2e'
    W[6, 0] = 'FB_l5e_l2i'
    W[6, 1] = 'FB_l5e_l2i'
    W[6, 2] = 'FB_l5e_l5e'
    W[6, 3] = 'FB_l5e_l5i'

    # Set the connections between regions
    # TODO: continue filling up this list
    W[0, 5] = 0 #region1_region2_l2e-l2i
    W[1, 4] = 0 # region2_region1 l2e-l2i
    W[0, 6] = 0 # region1 region2 l2e-l5e
    W[2, 4] = 0 # region2 region1 l2e-l5e
    W[0, 7] = 0 # region1 region2 l2e-l5i


    net.synapses.append(Synapse(id='rs',
                                lems_source_file='Prototypes.xml'))


    # Background input
    # Iterate over the different possible areas
    pops = []
    area_edge = 30
    area_spacing = 30
    
    layer_thickness = 30
    for area_idx, area in enumerate(areas):
        # Add populations
        x_offset = area_idx*(area_edge + area_spacing)
        l23 = RectangularRegion(id='%s_L23' %(area), x=x_offset, y=layer_thickness, z=0, width=area_edge, height=30, depth=area_edge)
        net.regions.append(l23)
        l56 = RectangularRegion(id='%s_L56' %(area), x=x_offset, y=0, z=0, width=area_edge, height=layer_thickness, depth=area_edge)
        net.regions.append(l56)

        pl23e = Population(id='%s_L23_E' %(area), size=1, component=l23ecell.id, properties={'color':color_str['l23e']},random_layout = RandomLayout(region=l23.id))
        pops.append(pl23e)
        pl23i = Population(id='%s_L23_I' %(area), size=1, component=l23icell.id, properties={'color':color_str['l23i']},random_layout = RandomLayout(region=l23.id))
        pops.append(pl23i)

        pl56e = Population(id='%s_L56_E' %(area), size=1, component=l56ecell.id, properties={'color':color_str['l56e']},random_layout = RandomLayout(region=l56.id))
        pops.append(pl56e)
        pl56i = Population(id='%s_L56_I' %(area), size=1, component=l56icell.id, properties={'color':color_str['l56i']},random_layout = RandomLayout(region=l56.id))
        pops.append(pl56i)

        net.populations.append(pl23e)
        net.populations.append(pl23i)

        net.populations.append(pl56e)
        net.populations.append(pl56i)

        # Add inputs
        input_source_l23 = InputSource(id='iclamp_23_%s' %area,
                                       neuroml2_input='PulseGenerator',
                                       parameters={'amplitude':'%snA'%Iext[area_idx][0], 'delay':'0ms', 'duration':'%sms'%duration})
        net.input_sources.append(input_source_l23)
        # Add modulation
        net.inputs.append(Input(id='modulation_l23_E',
                                input_source=input_source_l23.id,
                                population=pl23e.id,
                                percentage=100))
        input_source_l56 = InputSource(id='iclamp_56_%s' %area,
                                       neuroml2_input='PulseGenerator',
                                       parameters={'amplitude':'%snA'%Iext[area_idx][1], 'delay':'0ms', 'duration':'%sms'%duration})

        net.input_sources.append(input_source_l56)
        # Add modulation
        net.inputs.append(Input(id='modulation_l56_E',
                                input_source=input_source_l56.id,
                                population=pl56e.id,
                                percentage=100))

    for pre_pop in pops:
        for post_pop in pops:
            internal_connections(pops, W, pre_pop, post_pop)



    print(net)
    print(net.to_json())
    new_file = net.to_json_file('%s.json'%net.id)


    ################################################################################
    ###   Build Simulation object & save as JSON

    sim = Simulation(id='Sim%s_%d'%(net.id,count),
                     network=new_file,
                     duration=duration,
                     dt=dt,
                     seed=count,
                     recordRates={'all':'*'})

    sim.to_json_file()

    return sim, net

if __name__ == "__main__":

    from neuromllite.NetworkGenerator import check_to_generate_or_run
    from neuromllite.sweep.ParameterSweep import *
    from pyneuroml import pynml

    JEE = 1.5
    JIE = 3.5
    JEI = -3.25
    JII = -2.5

    pop_colors = {'L23_E':'#dd7777','L23_I':'#7777dd','L23_E Py':'#990000','L23_I Py':'#000099',
                  'L56_E':'#77dd77','L56_I':'#dd77dd','L56_E Py':'#009900','L56_I Py':'#990099'}

    if '-sweep' in sys.argv:
        # To do...
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
                        pop_type = pop[pop.index('_')+1:]
                        pop_color = pop_colors[pop_type]
                        colors.append(pop_color)

                        hist1, edges1 = np.histogram(traces[tr], bins=hist_bins)
                        mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                        histxs.append(mid1)
                        histys.append(hist1)
                        histcolors.append(pop_color)
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

                    hist1, edges1 = np.histogram(l23e, bins=hist_bins)
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

                    hist1, edges1 = np.histogram(l23i, bins=hist_bins)
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

                    hist1, edges1 = np.histogram(l56e, bins=hist_bins)
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

                    hist1, edges1 = np.histogram(l56i, bins=hist_bins)
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
                                linestyles=[('-' if 'Py' in l else '--') for l in labels],
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

        from intralaminar import intralaminar_analysis, intralaminar_plt

        wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = 0; l2e_l5e = 0
        # Input strength of the excitatory population

        # When the analysis argument is not passed run the intralaminar simulation for only one case
        # total number of simulations to run for each input strength
        if '-analysis' in sys.argv:
            Iexts = [0, 2, 4, 6]
            nruns = 10
        else:
            Iexts = [2]
            nruns = 1

        simulation = {}

        for Iext in Iexts:
            simulation[Iext] = {}
            for run in range(nruns):
                sim, net = generate(wee=wee, wei=wei, wie=wie, wii=wii, i_l5e_l2i=l5e_l2i, i_l2e_l5e=l2e_l5e, duration=25000,
                                    areas=['V1'], Iext=[[Iext, Iext]], count=run,
                                    net_id='Intralaminar')
                                    
                ################################################################################
                ###   Run in some simulators

                if not '-analysis' in sys.argv:
                    check_to_generate_or_run(sys.argv, sim)
                    
                else:
                    simulator = 'jNeuroML'

                    nmllr = NeuroMLliteRunner('%s.json'%sim.id,
                                              simulator=simulator)

                    traces, events = nmllr.run_once('/tmp')

                    simulation[Iext][run] = {}
                    # For the purpose of this analysis we will save only the traces related to the excitatory L23 population
                    simulation[Iext][run]['L23_E/0/L23_E/r'] = np.array(traces['V1_L23_E/0/L23_E_comp/r'])


        if '-analysis' in sys.argv:
            # analyse the traces using python methods
            psd_dic = intralaminar_analysis(simulation, Iexts, nruns, layer='L23', dt=2e-04, transient=5)
            # plot the results
            intralaminar_plt(psd_dic)


    elif '-interlaminar' in sys.argv:
        import matplotlib.pylab as plt

        from interlaminar import calculate_interlaminar_power_spectrum, plot_interlaminar_power_spectrum, \
                                 plot_power_spectrum_neurodsp

        # Load the python results (this script assumes that the python script
        # Mejias-2016.py -interlaminar_a has already
        #  generated the pickle file with the results).
        simulation_file = '../Python/debug/interlaminar_a/simulation.pckl'
        with open(simulation_file, 'rb') as filename:
            pyrate = pickle.load(filename)

        dt = 2e-01
        transient = 10
        Nbin = 100
        if '-analysis' in sys.argv:
            duration = 6e05
        else:
            duration = 1e03

        wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = .75; l2e_l5e = 1
        sim, net = generate(wee=wee, wei=wei, wie=wie, wii=wii, i_l5e_l2i=l5e_l2i, i_l2e_l5e=l2e_l5e, dt=dt,
                            areas=['V1'], duration=duration, Iext=[[8, 8]], count=0,
                            net_id='Interlaminar')
        # Run in some simulators
        check_to_generate_or_run(sys.argv, sim)


        if '-analysis' in sys.argv:
            
            simulator = 'jNeuroML'

            nmllr = NeuroMLliteRunner('%s.json' % sim.id,
                                      simulator=simulator)
            traces, events = nmllr.run_once('/tmp')
            
            rate_conn = np.stack((np.array(traces['V1_L23_E/0/L23_E_comp/r']),
                                  np.array(traces['V1_L23_I/0/L23_I_comp/r']),
                                  np.array(traces['V1_L56_E/0/L56_E_comp/r']),
                                  np.array(traces['V1_L56_I/0/L56_I_comp/r']),
                                  ))

            # for compatibility with the Python code, expand the third dimension
            rate_conn = np.expand_dims(rate_conn, axis=2)
            # TODO: Make this more obvious
            # transform the dt from ms to s, for the rest of the analysis
            s_dt = dt / 1000
            pxx_coupled_l23_bin, fxx_coupled_l23_bin, pxx_coupled_l56_bin, fxx_coupled_l56_bin = \
                    calculate_interlaminar_power_spectrum(rate_conn, s_dt, transient, Nbin)

            xs1 = []
            ys1 = []
            labels1 = []
            xs2 = []
            ys2 = []
            labels2 = []
            histxs = []
            histys = []
            histlabels = []
            colors = []
            histcolors = []
            hist_bins = 50
            pop_colors = {'V1_L23_E': '#dd7777', 'V1_L23_I': '#7777dd', 'L23_E_Py':'#990000','L23_I_Py':'#000099',
                          'V1_L56_E': '#77dd77', 'V1_L56_I': '#dd77dd', 'L56_E_Py':'#009900','L56_I_Py':'#990099'}

            # Append traces generated with NeuroML
            for tr in traces:
                if tr != 't':
                    xs1.append(traces['t'])
                    ys1.append(traces[tr])
                    pop = tr.split('/')[0]
                    labels1.append(pop)
                    colors.append(pop_colors[pop])

                    hist1, edges1 = np.histogram(traces[tr], bins=hist_bins)
                    mid1 = [e + (edges1[1] - edges1[0]) / 2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histcolors.append(pop_colors[pop])
                    histlabels.append(pop)

            # Append Python traces
            for key in pyrate:
                if key.endswith('/conn'):
                    xs2.append(pyrate['ts'])
                    ys2.append(pyrate[key])
                    pop = key.split('/')[0]
                    labels2.append(pop)
                    colors.append(pop_colors[pop])

                    hist1, edges1 = np.histogram(pyrate[key], bins=hist_bins)
                    mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histlabels.append(pop)
                    histcolors.append(pop_colors[pop])

            pynml.generate_plot(xs1,
                                ys1,
                                'With connections Rates',
                                show_plot_already=False,
                                labels=labels1,
                                linewidths=[(1 if 'Py' in l else 2) for l in labels1],
                                yaxis='Rate (Hz)',
                                xaxis='Time (s)',
                                legend_position='right',
                                title_above_plot=True)
            pynml.generate_plot(xs2,
                                ys2,
                                'With connections Rates',
                                show_plot_already=False,
                                labels=labels2,
                                linewidths=[(1 if 'Py' in l else 2) for l in labels2],
                                yaxis='Rate (Hz)',
                                xaxis='Time (s)',
                                legend_position='right',
                                title_above_plot=True)

            pynml.generate_plot(histxs,
                                histys,
                                'Histograms: With Connection',
                                labels=histlabels,
                                colors=histcolors,
                                show_plot_already=False,
                                xaxis='Rate bins (Hz)',
                                yaxis='Num timesteps rate in bins',
                                markers=['o' for x in histxs],
                                markersizes=[2 for x in histxs],
                                legend_position='right',
                                title_above_plot=True)


            # Repeat the calculations for the case where there is no connection between layers
            wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = 0; l2e_l5e = 0
            sim, net = generate(wee=wee, wei=wei, wie=wie, wii=wii, i_l5e_l2i=l5e_l2i, i_l2e_l5e=l2e_l5e, duration=duration,
                                areas=['V1'], Iext=[[8, 8]], count=0)
            # Run in some simulators
            check_to_generate_or_run(sys.argv, sim)
            simulator = 'jNeuroML'

            nmllr = NeuroMLliteRunner('%s.json' % sim.id,
                                      simulator=simulator)
            traces, events = nmllr.run_once('/tmp')
            rate_noconn = np.stack((np.array(traces['V1_L23_E/0/L23_E_comp/r']),
                                  np.array(traces['V1_L23_I/0/L23_I_comp/r']),
                                  np.array(traces['V1_L56_E/0/L56_E_comp/r']),
                                  np.array(traces['V1_L56_I/0/L56_I_comp/r']),
                                  ))
            # for compatibility with the Python code, expand the third dimension
            rate_noconn = np.expand_dims(rate_noconn, axis=2)

            xs1 = []
            ys1 = []
            labels1 = []
            xs2 = []
            ys2 = []
            labels2 = []
            histxs = []
            histys = []
            histlabels = []
            colors = []
            histcolors = []

            for tr in traces:
                if tr != 't':
                    xs1.append(traces['t'])
                    ys1.append(traces[tr])
                    pop = tr.split('/')[0]
                    labels1.append(pop)
                    colors.append(pop_colors[pop])

                    hist1, edges1 = np.histogram(traces[tr], bins=hist_bins)
                    mid1 = [e + (edges1[1] - edges1[0]) / 2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histcolors.append(pop_colors[pop])
                    histlabels.append(pop)

            # Append Python traces
            for key in pyrate:
                if key.endswith('/unconn'):
                    xs2.append(pyrate['ts'])
                    ys2.append(pyrate[key])
                    pop = key.split('/')[0]
                    labels2.append(pop)
                    colors.append(pop_colors[pop])

                    hist1, edges1 = np.histogram(pyrate[key], bins=hist_bins)
                    mid1 = [e +(edges1[1]-edges1[0])/2 for e in edges1[:-1]]
                    histxs.append(mid1)
                    histys.append(hist1)
                    histlabels.append(pop)
                    histcolors.append(pop_colors[pop])


            pynml.generate_plot(xs1,
                                ys1,
                                'No connections Rates',
                                show_plot_already=False,
                                labels=labels1,
                                linewidths=[(1 if 'Py' in l else 2) for l in labels1],
                                yaxis='Rate (Hz)',
                                xaxis='Time (s)',
                                legend_position='right',
                                title_above_plot=True)
            pynml.generate_plot(xs2,
                                ys2,
                                'No connections Rates',
                                show_plot_already=False,
                                labels=labels2,
                                linewidths=[(1 if 'Py' in l else 2) for l in labels2],
                                yaxis='Rate (Hz)',
                                xaxis='Time (s)',
                                legend_position='right',
                                title_above_plot=True)

            pynml.generate_plot(histxs,
                                histys,
                                'Histograms: No Connection',
                                labels=histlabels,
                                colors=histcolors,
                                show_plot_already=False,
                                xaxis='Rate bins (Hz)',
                                yaxis='Num timesteps rate in bins',
                                markers=['o' for x in histxs],
                                markersizes=[2 for x in histxs],
                                legend_position='right',
                                title_above_plot=True)

            pxx_uncoupled_l23_bin, fxx_uncoupled_l23_bin, pxx_uncoupled_l56_bin, fxx_uncoupled_l56_bin = \
                calculate_interlaminar_power_spectrum(rate_noconn, s_dt, transient, Nbin)

            # Plot the Power Spectrum Analysis
            plot_power_spectrum_neurodsp(s_dt, rate_conn, rate_noconn, 'interlaminar')

            # Plot spectrogram
            plot_interlaminar_power_spectrum(fxx_uncoupled_l23_bin, fxx_coupled_l23_bin,
                                             pxx_uncoupled_l23_bin, pxx_coupled_l23_bin,
                                             fxx_uncoupled_l56_bin, fxx_coupled_l56_bin,
                                             pxx_uncoupled_l56_bin, pxx_coupled_l56_bin,
                                             'interlaminar')

            plt.show()
            
    elif '-interareal' in sys.argv:
        from interareal import interareal_analysis, interareal_plt, load_sln_data



        # Background current simulation.
        # Note: For testing purpose, only the rest simulation is performed if the flag '-analysis' is not
        # passed
        wee = JEE; wei = JIE; wie = JEI; wii = JII; l5e_l2i = .75; l2e_l5e = 1
        FF_l2e_l2e = 1; FB_l5e_l2i = .5; FB_l5e_l5e=.9; FB_l5e_l5i = .5; FB_l5e_l2e = .1
        dt = 2e-01
        transient = 5
        duration = 4e04
        minfreq_l23 = 30. # Hz
        minfreq_l56 = 3. # Hz
        areas = ['V1', 'V4']
        nareas = len(areas)

        # Load previously calculated SLN to define which regions are FF and which FB
        sln_file_path = 'sln.csv'
        df_sln = pd.read_csv('sln.text', index_col=0)

        # Find SLN values for the areas under analysis
        # First, check if we have the SLN values for all areas passed for the analysis

        missing_area = [area for area in areas if area not in df_sln.columns]
        assert(len(missing_area) > 0, 'Missing SLN values for the passed list')
        sln = []
        max_sln = 0
        for region1, region2 in permutations(areas, 2):
            print(region1 + '-->' + region2 + ': ' + str(df_sln[region1][region2]))
            if df_sln[region1][region2] > max_sln:
                max_sln = df_sln[region1][region2]
                # TODO: you might want to have a list of lists
                ff_connection = [region1, region2]
        for connection in range(len(ff_connection)):
            print('FF connection between: %s -> %s' %(ff_connection[connection][0], ff_connection[connection][1]))

        # for testing purpose generate one single simulation
        if '-analysis' in sys.argv:
            stats = 10
        else:
            stats = 1

        if '-stimulate_V1' in sys.argv:
            # Iext0: Excitatory current on E L23,
            # Iext1: Excitatory current on layer E l56
            # The first 2 values correspond to the first area, the last 2 values for the second area
            stimulated_area = 'stimulate_V1'
            Iext0 = 2; Iext1 = 4 # background current at excitatory population
            Iext_rest = [[Iext0, Iext1],[Iext0, Iext1]]
            # Injection is applied at V1
            stim = 15
            Iext_stim = [[stim + Iext0, stim + Iext1], [Iext0, Iext1]]
        elif '-stimulate_V4' in sys.argv:
            stimulated_area = 'stimulate_V4'
            Iext0 = 1; Iext1= 1 # background current at excitatory population
            Iext_rest = [[Iext0, Iext1],[Iext0, Iext1]]
            # Injection is applied at V4
            stim = 15
            Iext_stim = [[Iext0, Iext1], [stim + Iext0, stim + Iext1]]
        else:
            raise IOError('Please make sure you are stimulating either V1 (-stimulate_V1) or V4 (-stimulate_V4). Stimulation to other areas are not yet supported.')

        traces_rest_stats = []
        traces_stim_stats = []
        for stat in range(stats):
            # Run simulation at rest
            sim_rest, net_rest = generate(wee=wee, wei=wei, wie=wie, wii=wii,
                                          i_l5e_l2i=l5e_l2i, i_l2e_l5e=l2e_l5e,
                                          areas=areas, FF_l2e_l2e=FF_l2e_l2e, FB_l5e_l2i=FB_l5e_l2i, FB_l5e_l5e=FB_l5e_l5e,
                                          FB_l5e_l5i=FB_l5e_l5i, FB_l5e_l2e=FB_l5e_l2e,
                                          dt=dt, duration=duration, Iext=Iext_rest, count=stat,
                                          net_id='Interareal')
            # Run in some simulators
            check_to_generate_or_run(sys.argv, sim_rest)
            simulator = 'jNeuroML'

            nmllr_rest = NeuroMLliteRunner('%s.json' %sim_rest.id,
                                           simulator=simulator)
            traces_rest, events_rest = nmllr_rest.run_once('/tmp')
            # if -analysis is not passed generate a single network using the rest as default

            # Run simulation with microsimulation
            sim_stim, net_stim = generate(wee=wee, wei=wei, wie=wie, wii=wii,
                                          i_l5e_l2i=l5e_l2i, i_l2e_l5e=l2e_l5e,
                                          areas=areas, FF_l2e_l2e=FF_l2e_l2e, FB_l5e_l2i=FB_l5e_l2i, FB_l5e_l5e=FB_l5e_l5e,
                                          FB_l5e_l5i=FB_l5e_l5i, FB_l5e_l2e=FB_l5e_l2e,
                                          dt=dt, duration=duration, Iext=Iext_stim, count=stat)
            # Run in some simulators
            check_to_generate_or_run(sys.argv, sim_stim)
            simulator = 'jNeuroML'

            nmllr_stim = NeuroMLliteRunner('%s.json' %sim_stim.id,
                                           simulator=simulator)
            traces_stim, events_stim = nmllr_stim.run_once('/tmp')



            # Save the traces for each different run
            traces_rest_stats.append(traces_rest)
            traces_stim_stats.append(traces_stim)

        if '-analysis' in sys.argv:

            # Concatenate the simulation results in the same format as the matlab code
            # No stimulus
            rate_rest_all = np.zeros((len(traces_rest_stats[0]) - 1, len(traces_rest_stats[0]['V1_L23_E/0/L23_E_comp/r']), stats))
            for stat in range(stats):
                for idx, key in enumerate(sorted(traces_rest_stats[stat].keys())):
                    if key is not 't':
                        rate_rest_all[idx, :, stat] = traces_rest_stats[stat][key]
            # With stimulus
            rate_stim_all = np.zeros((len(traces_rest_stats[0]) - 1, len(traces_rest_stats[0]['V1_L23_E/0/L23_E_comp/r']), stats))
            for stat in range(stats):
                for idx, key in enumerate(sorted(traces_stim_stats[stat].keys())):
                    if key is not 't':
                        rate_stim_all[idx, :, stat] = traces_stim_stats[stat][key]

            # TODO: Make this more obvious
            # for compatibility with the Python code, expand the third dimension
            rate_rest = np.expand_dims(rate_rest_all, axis=2)
            rate_stim = np.expand_dims(rate_stim_all, axis=2)
            # transform the dt from ms to s, for the rest of the analysis
            s_dt = dt / 1000
            # perform simulation with background current

            #rate_stim = np.stack([traces_stim[key] for key in sorted(traces_stim.keys()) if key is not 't'])
            # TODO: Make this more obvious

            # Perfrom analysis with both rest and stimulated simulation
            px20, px2, px50, px5, fx2, pgamma, palpha = interareal_analysis(rate_rest, rate_stim, transient, s_dt, minfreq_l23, minfreq_l56,
                                                            nareas, stats)

            # Plot the results
            interareal_plt(areas, px20, px2, px50, px5, fx2, stimulated_area, stats)

        print('Done')


    else:

        sim, net = generate()

        ################################################################################
        ###   Run in some simulators


        check_to_generate_or_run(sys.argv, sim)

