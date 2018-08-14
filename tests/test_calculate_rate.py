#!/usr/bin/env python
import numpy as np
import pickle
import unittest
from deepdiff import DeepDiff
# set random seed
np.random.RandomState(seed=42)

from Python.calculate_rate import calculate_rate

class TestCalculateRate(unittest.TestCase):

    def setUp(self):
        # path to the golden results
        self.golden = 'tests/golden/simulation.pckl'

        # settings for the model
        self.dt = 2e-4
        self.tstop = 1
        self.t = np.arange(0, self.tstop, self.dt)
        self.transient = 0
        # Connection between layers
        wee = 1.5; wei = -3.25
        wie = 3.5; wii = -2.5
        # define interlaminar synaptic coupling strenghts
        J_2e = 0; J_2i = 0
        J_5e = 0; J_5i = 0
        self.J = np.array([[wee, wei, J_5e, 0],
                           [wie, wii, J_5i, 0],
                           [J_2e, 0, wee, wei],
                           [J_2i, 0, wie, wii]])
        # Specify membrane time constants
        tau_2e = 0.006; tau_2i = 0.015
        tau_5e = 0.030; tau_5i = 0.075
        self.tau = np.array([[tau_2e], [tau_2i], [tau_5e], [tau_5i]])

        # sigma
        sig_2e = .3; sig_2i = .3
        sig_5e = .45; sig_5i = .45
        self.sig = np.array([[sig_2e], [sig_2i], [sig_5e], [sig_5i]])

        self.Iexts = [0]
        self.Ibgk = np.zeros((self.J.shape[0], 1))
        self.nruns = [1]
        self.noise = 1
        self.Nareas = 1

    def test_calculate_rate(self):
        rate = calculate_rate(self.t, self.dt, self.tstop, self.J, self.tau,
                              self.sig, self.Iexts, self.Ibgk, self.noise,
                              self.Nareas)
        # compare results with the golden truth
        with open(self.golden, 'rb') as fh:
            golden_results = pickle.load(fh)
        self.assertEqual(DeepDiff(rate, golden_results), {})

if __name__ == '__main__':
    unittest.main()
