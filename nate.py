'''
stacking for nebular lines
'''

from importer import *
import marvin

import matplotlib.pyplot as plt
import numpy as np

import bins, ndice

# define binning space
bin_logSMSD = bins.BinDef('logSMSD', np.linspace(0., 3., 10), 0) # Msun/pc2
bin_BPT = bins.BinDef('BPTclass', [.5, 1.5, 2.5, 3.5, 4.5], 1) # BPT [None, SF, LI(N)ER, AGN, Comp.]
bin_ewHa = bins.BinDef('ewHa', [3., 8., 14.], 2) # EW of Ha emission (DIG proxy--Lacerda+17)
bin_OiiiOii = bins.BinDef('OiiiOii', [2., 5., 8., 11.], 3) # ionization parameter sensitive oiii5007/oii3727
bin_tauV = bins.BinDef('tauV', [.2, .4, .6, .8, 1., 1.3, 1.6, 2., 2.5])

binner = bins.Binner([bin_logSMSD, bin_BPT, bin_ewHa, bin_OiiiOii, bin_tauV])

