import os, sys, matplotlib
import warnings

#####
# astropy cosmology science state
#####

from astropy.cosmology import WMAP9
cosmo = WMAP9

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = True

#####
# marvin configuration
#####

MPL_V = 'MPL-8'
DAPBINTYPE = 'SPX'
DAPTEMPLATETYPE = 'MILESHC-MILESHC'
DAPTYPE = '-'.join((DAPBINTYPE, DAPTEMPLATETYPE))
SAS_BASEDIR = os.environ['SAS_BASE_DIR']

STACKING_BASEDIR = '/usr/data/minhas2/zpace/stacking'

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    from marvin import config as marvin_cfg
    
    marvin_cfg.release = MPL_V
    marvin_cfg.download = True
    marvin_cfg.access = 'collab'
    MANGADRPVER, MANGADAPVER = marvin_cfg.lookUpVersions()

    print('----Marvin Configuration-----')
    print(''.join(('\t', 'Release: {}'.format(marvin_cfg.release))))
    print(''.join(('\t', '\t', 'DRP: {}'.format(MANGADRPVER))))
    print(''.join(('\t', '\t', 'DAP: {}'.format(MANGADAPVER))))
    print(''.join(('\t', '\t', '\t', 'Bin Type: {}'.format(DAPBINTYPE))))
    print(''.join(('\t', '\t', '\t', 'Template Type: {}'.format(DAPTEMPLATETYPE))))
    print(''.join(('\t', 'Local SAS: {}'.format(SAS_BASEDIR))))
    print(''.join(('\t', 'Download: {}'.format(marvin_cfg.download))))
    print(''.join(('\t', 'Access: {}'.format(marvin_cfg.access))))
