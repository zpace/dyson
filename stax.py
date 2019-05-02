import marvin

import numpy as np
import matplotlib.pyplot as plt

from speclite import redshift, accumulate

from astropy.io import fits
from astropy.utils.decorators import lazyproperty

import dataclasses
from collections import deque
import os


@dataclasses.dataclass
class LogWaveGrid(object):
    lamc0: float
    nl: int
    dlogl: float

    @lazyproperty
    def loglamc(self):
        '''centers of log-wavelength channels
        '''
        return np.log10(self.lamc0) + self.dlogl * np.linspace(0., self.nl - 1, self.nl)

    @lazyproperty
    def loglaml(self):
        return self.loglamc - 0.5 * self.dlogl

    @lazyproperty
    def loglamu(self):
        return self.loglamc + 0.5 * self.dlogl

    @lazyproperty
    def lamc(self):
        '''centers of wavelength channels
        '''
        return 10.**self.loglamc

    @lazyproperty
    def laml(self):
        return 10.**self.loglaml

    @lazyproperty
    def lamu(self):
        return 10.**self.loglamu


class BinStack(object):
    def __init__(self, lamgrid, bin_ix, binnum):
        self.lamgrid = lamgrid
        self.data_accum = None
        self.z_accum = deque()
        self.bin_ix = bin_ix
        self.binnum = binnum

    def spaxels_in_bin(self, assign_hdu):
        '''evaluate which spaxels reside in current bin
        '''
        bin_assignments = assign_hdu['ASSIGN'].data
        mask = assign_hdu['MASK'].data.astype(bool)

        goes_in_this_bin = np.all(
            bin_assignments == self.bin_ix[:, None, None], axis=0) * ~mask

        return goes_in_this_bin

    def incorporate_galaxy(self, plateifu, path_to_assign, elresids, z_map):
        '''incorporate a galaxy into the bin stack
        '''
        with fits.open(path_to_assign) as assign_hdu:
            goes_in_this_bin = self.spaxels_in_bin(assign_hdu)
        
        # are there any matching spaxels in this galaxy?
        # if yes, process them
        if goes_in_this_bin.sum() > 0:
            for i, j in zip(*np.where(goes_in_this_bin)):
                self.data_accum = accumulate(
                    data1_in=self.data_accum, data2_in=elresids[:, i, j],
                    data_out=self.data_accum, add='diffs', weight='ivars')

            self.z_accum.extend(z_map[goes_in_this_bin])

        # if no, proceed
        else:
            pass

    def plot(self, *args, **kwargs):
        plt.plot(self.lamgrid.lamc, self.data_accum['diffs'], *args, **kwargs)

        return plt.gcf()

    def write(self, loc='.'):
        '''write out results to a file
        '''
        binnum = np.ravel_multi_index

        hdulist = fits.HDUList(fits.PrimaryHDU())

        for i, ax_ix in enumerate(self.bin_ix):
            hdulist[0].header['IX_AX{}'.format(i)] = ax_ix

        lam_hdu = fits.ImageHDU(self.lamgrid.lamc)
        lam_hdu.header['EXTNAME'] = 'LAM'
        hdulist.append(lam_hdu)

        resid_hdu = fits.ImageHDU(self.data_accum['diffs'])
        resid_hdu.header['EXTNAME'] = 'RESID'
        hdulist.append(resid_hdu)

        ivar_hdu = fits.ImageHDU(self.lamgrid.lamc)
        ivar_hdu.header['EXTNAME'] = 'IVAR'
        hdulist.append(ivar_hdu)

        z_hdu = fits.ImageHDU(np.array(self.z_accum))
        z_hdu.header['EXTNAME'] = 'ZS'
        hdulist.append(z_hdu)

        hdulist.writeto(os.path.join(loc, '{}_stack.fits'.format(self.binnum)),
                        overwrite=True)
