'''
stacking for nebular lines
'''

from importer import *
import marvin

from astropy import units as u, constants as c, table as t
from astropy.utils.console import ProgressBar

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from astropy.io import fits

import bins
import stax
import drizzle


def rect_spec_minus_stelcont(plateifu, z_map, wavegrid):
    '''evaluates the difference between the observed spectrum and the stellar continuum fit

    Parameters
    ----------
    plateifu : str
        plateifu galaxy designation

    z_map : np.ndarray
        array of spaxel redshifts

    wavegrid : `stax.LogWaveGrid` instance
        wavelength grid with identical spacing in log-wavelength
    '''
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dap_logcube = marvin.tools.ModelCube(plateifu, bintype=DAPBINTYPE, template=DAPTEMPLATETYPE)
        drp_logcube = marvin.tools.Cube(plateifu)

    diff = drp_logcube.flux.masked - dap_logcube.stellarcont_fit.value
    ivar = drp_logcube.flux.ivar
    lam = drp_logcube.flux.wavelength.value[:, None, None]

    diff = diff * (1. + z_map)[None, ...]**-1
    ivar = ivar * (1. + z_map)[None, ...]**2.
    lam = lam * (1. + z_map)[None, ...]**-1

    # drizzle them into the rest-frame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rect_diff, rect_var = drizzle.drizzle_mangalogcube(
            oldlcube_c=lam, oldfluxcube=diff, oldvarcube=1. / ivar, newl_c=wavegrid.lamc,
            dlogl=wavegrid.dlogl)

    resid = np.ma.array(
        np.zeros_like(rect_diff.data), dtype=[('diffs', float), ('ivars', float)])
    resid['diffs'] = rect_diff
    resid['ivars'] = 1. / rect_var

    return resid


if __name__ == '__main__':

    # setup
    basedir = STACKING_BASEDIR
    assign_basedir = os.path.join(basedir, 'assign')
    binstacks_basedir = os.path.join(basedir, 'binstacks')

    loglgrid = stax.LogWaveGrid(lamc0=3800., dlogl=1.0e-4, nl=3700)

    # load drpall and dapall, index by plateifu
    drpall = marvin.utils.general.general.get_drpall_table()
    drpall.add_index('plateifu')

    dapall = t.Table.read(
        marvin.utils.general.general.get_dapall_file(MANGADRPVER, MANGADAPVER))
    dapall = dapall[dapall['DAPTYPE'] == '{}-{}'.format(DAPBINTYPE, DAPTEMPLATETYPE)]
    dapall = dapall[dapall['DAPDONE']]
    dapall.add_index('PLATEIFU')

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
    binner.define_binstacks(loglgrid)

    #'''
    for drpall_row in ProgressBar(drpall):
        plateifu = drpall_row['plateifu']
        if not ([plateifu] in dapall['PLATEIFU']):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dap_maps = marvin.tools.Maps(plateifu, bintype=DAPBINTYPE, template=DAPTEMPLATETYPE)
        bin_assign = binner([dap_maps] * 4)

        bins.write_bin_assign(bin_assign, plateifu, assign_basedir)
    #'''

    '''
    # loop through galaxies
    for plateifu in plateifu_list:
        path_to_assign = os.path.join(assign_basedir, '{}_BINASSIGN.fits'.format(plateifu))
        z_map = (1. * dap_maps.nsa.z) * \
                (1. + (dap_maps.emline_gvel_ha_6564 / c.c).decompose().value)
        elresids = rect_spec_minus_stelcont(plateifu, z_map, loglgrid)

        for bin_row, bin_stack in zip(binner.table, binner.bin_stacks):
            bin_stack.incorporate_galaxy(plateifu, path_to_assign, elresids, z_map)
    '''

    