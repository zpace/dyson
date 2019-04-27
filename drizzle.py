import numpy as np
from scipy import sparse
import utils as ut

import numba

@numba.jit
def drizzle_mangalogcube(oldlcube_c, oldfluxcube, oldvarcube, newl_c, dlogl):
    '''
    drizzle cube from an old wavelength grid to a new
    '''
    oldcubeshape = oldlcube_c.shape
    mapshape = oldcubeshape[1:]
    newnl = len(newl_c)
    newcubeshape = (newnl, ) + mapshape

    newfluxcube, newvarcube = np.zeros(newcubeshape), np.zeros(newcubeshape)

    # make old log wavelength cube, and l+r edges
    oldloglcube_c = np.log10(oldlcube_c)
    oldloglcube_le = oldloglcube_c - 0.5 * dlogl
    oldloglcube_ue = oldloglcube_c + 0.5 * dlogl
    oldloglcube_e = np.concatenate([oldloglcube_le, oldloglcube_ue[-1:]])
    
    oldlcube_le = 10.**oldloglcube_le
    oldlcube_ue = 10.**oldloglcube_ue
    oldlcube_e = np.concatenate([oldlcube_le, oldlcube_ue[-1:]])

    # make new log wavelength array, and l+r edges
    newlogl_c = np.log10(newl_c)
    newlogl_le = newlogl_c - 0.5 * dlogl
    newlogl_ue = newlogl_c + 0.5 * dlogl

    newl_le = 10.**newlogl_le
    newl_ue = 10.**newlogl_ue

    newl_e = np.concatenate([newl_le, newl_ue[-1:]])

    for i, j in np.ndindex(*mapshape):
        il0, fl = _drizzle_info_samegridscale(oldlcube_e[:, i, j], newl_e)
        iu0, fu = il0 + 1, 1. - fl

        newflux = fl * oldfluxcube[il0:il0 + newnl, i, j] + \
                  fu * oldfluxcube[iu0:iu0 + newnl, i, j]

        newvar = fl * oldvarcube[il0:il0 + newnl, i, j] + \
                 fu * oldvarcube[iu0:iu0 + newnl, i, j]

        newfluxcube[:, i, j] = newflux
        newvarcube[:, i, j] = newvar

    return newfluxcube, newvarcube

@numba.jit
def _drizzle_info_samegridscale(lam_edges_old, lam_edges_new):
    '''compute transformation between wavelength scales, assuming same wavelength scale
    '''

    # find smallest bin edge in old that's larger than the smallest bin edge in new
    for oldix, (oldle, oldre) in enumerate(zip(lam_edges_old[:-1], lam_edges_old[1:])):
        if (oldle < lam_edges_new[0]) and (lam_edges_new[0] <= oldre):
            oldre_startval = oldre
            oldle_startval = oldle
            oldedges_startix_le = oldix
            break
        else:
            continue

    # weight given to lower wavelength solution
    new_frac_from_lneighbor = (oldre_startval - lam_edges_new[0]) / \
                              (lam_edges_new[1] - lam_edges_new[0])
    new_frac_from_uneighbor = 1. - new_frac_from_lneighbor

    return oldedges_startix_le, new_frac_from_lneighbor



if __name__ == '__main__':
    dlogl = 1.0e-4
    old_lamlims = [3600., 10400.]
    new_lamlims = [3800., 8900.]
    NI, NJ = 74, 74

    zs = .05 + .001 * np.random.randn(NI, NJ)
    old_loglam_edges = np.arange(*list(map(np.log10, old_lamlims)), dlogl)[:, None, None] + \
                       zs / np.log10(np.e)
    old_lam_edges = 10.**old_loglam_edges
    old_lam_ctrs = 0.5 * (old_lam_edges[1:, ...] + old_lam_edges[:-1, ...])

    new_loglam_edges = np.arange(*list(map(np.log10, new_lamlims)), dlogl)
    new_lam_edges = 10.**new_loglam_edges
    new_lam_ctrs = 0.5 * (new_lam_edges[1:] + new_lam_edges[:-1])

    flam_old = np.tile(np.ones(len(old_lam_edges) - 1)[:, None, None], (1, NI, NJ))
    flamvar_old = np.tile(np.ones(len(old_lam_edges) - 1)[:, None, None], (1, NI, NJ))
    drizzle_mangalogcube(oldlcube_c=old_lam_ctrs, oldfluxcube=flam_old,
                         oldvarcube=flamvar_old, newl_c=new_lam_ctrs, dlogl=dlogl)