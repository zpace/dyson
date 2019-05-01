'''
stacking for nebular lines
'''

from importer import *
import marvin

import matplotlib.pyplot as plt
import numpy as np

import bins, ndice

# define binning space

bin_BPT = bins.BPTBinMarvin('BPTclass', 0) # BPT class [SF, comp, seyfert, liner]

fn_ewHa = lambda marvin_maps: marvin_maps.getMap('emline_sew', 'ha_6564')
bin_ewHa = bins.BinDef('ewHa', [1.0e-5, 3., 8., 14., np.inf], 1, fn=fn_ewHa) # EW of Ha emission (DIG proxy--Lacerda+17)

fn_OiiiOii = lambda marvin_maps: marvin_maps.getMapRatio('emline_gflux', 'oiii_5007', 'oii_3727')
bin_OiiiOii = bins.BinDef('OiiiOii', np.logspace(-1., 2, 10), 2, fn=fn_OiiiOii) # ionization parameter sensitive oiii5007/oii3727

fn_BaDec = lambda marvin_maps: marvin_maps.getMapRatio('emline_gflux', 'ha_6563', 'hb_4861')
bin_BaDec = bins.BinDef('HaHb', np.arange(3.25, 6., .25), 3, fn=fn_BaDec)

fn_Rreff = lambda marvin_maps: marvin_maps.bin_lwellcoo_r_re
bin_Rreff = bins.BinDef('Rreff', np.array([0., 0.25, 0.5, 0.75, 1., 1.5, 2., 2.5, 3., 4., np.inf]), 4, fn=fn_Rreff)

binner = bins.Binner([bin_BPT, bin_ewHa, bin_OiiiOii, bin_Rreff])

plateifu_list = []