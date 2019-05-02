import numpy as np
from astropy import table as t

from astropy.io import fits
from astropy.utils.decorators import lazyproperty

import os

import utils as ut

from stax import BinStack

class BinDef(object):    
    def __init__(self, name, binbounds, index, *args, 
                 fn=None, **kwargs):
        '''defines the bins for a single quantity
        
        bins a single quantity according to a set of boundaries
        
        Parameters
        ----------
        name : str
            short description of quantity to be binned
        binbounds : array-like
            array of bin boundaries: all values less than `binbounds[0]` get placed
            into bin zero; all values between `binbounds[0]` and `binbounds[1]`
            get placed in bin one; etc.
        index : int
            index of bin, to be used when assembling bins of different quantities,
            to preserve order
        fn : {function}, optional
            function which acts on values passed to the binner, 
            with extra arguments `*args` and `**kwargs`, before binning takes place
            (the default is None, which implies no change to values passed)
        *args : tuple
            extra arguments, to be passed to `fn`
        **kwargs : dict
            extra keyword arguments, to be passed to `fn`

        Examples
        --------
        >>> a = np.linspace(1, 10, 100).reshape((10, 10))
        >>> tfm = lambda x, y: np.log10(x) + y
        >>> bin = BinDef('log_a', index=0, binbounds=[0.1, 0.5, 1.], fn=tfm, y=-0.1)
        >>> bin(a)
        '''
        self.name = name
        self.binbounds = np.asarray(binbounds)
        self.nbins = len(self.binbounds) - 1
        self.index = index

        if fn is None:
            self.fn = lambda x, *args, **kwargs: x
        else:
            self.fn = fn

        self.fn_args, self.fn_kwargs = args, kwargs

    def do_bin(self, tfm_vals):
        '''simply applies bin function
        '''
        assignment = np.digitize(tfm_vals, bins=self.binbounds)
        masked_assignment = np.ma.masked_outside(
            assignment, 1, len(self.binbounds) - 1)

        return masked_assignment

    def __call__(self, vals):
        '''applies transformation to passed values, then bins result
        
        transforms `vals` with `self.fn`, with extra arguments `self.fn_args`
        and keyword arguments `self.fn_kwargs`, and then bins the resulting values 
        according to `self.binbounds`
        
        Parameters
        ----------
        vals : array_like
            values to transform and then bin
        
        Returns
        -------
        array-like
            array of bin assignments
        '''
        assignment = self.do_bin(self.fn(vals, *self.fn_args, **self.fn_kwargs))
        return assignment

    @property
    def table(self):
        '''table of bin boundaries
        
        Returns
        -------
        astropy.table.Table
            table of bin boundaries
        '''
        index_col_name = '{}_index'.format(self.name)
        tab = t.Table(data=[self.binbounds[:-1], self.binbounds[1:], range(self.nbins)],
                      names=['{}_low'.format(self.name), '{}_high'.format(self.name), 
                             index_col_name])
        tab.add_index(index_col_name)
        return tab


class TableBinDef(BinDef):
    def __init__(self, tab, ix_colname, qty_colname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = dict(zip(tab[ix_colname], tab[qty_colname]))

    def __call__(self, ix):
        return np.digitize(self.fn(self.mapping[ix], *self.fn_args, **self.fn_kwargs), bins=self.binbounds)

class BPTBinMarvin(BinDef):
    '''
    defines a bin based on ionization diagnostic
    '''
    def __init__(self, name, index, *args, categories=['sf', 'comp', 'seyfert', 'liner'],
                 **kwargs):

        self.categories = categories
        binbounds = np.array(list(range(len(categories) + 1))) - 0.5

        super().__init__(name, binbounds, index, *args, fn=self.bpt, **kwargs)

    def bpt(self, marvin_dap_maps):
        '''function that classifies spaxels

        Parameters
        ----------
        marvin_dap_maps : marvin.tools.maps.Maps
            `marvin` `Maps` object
        '''
        bptmap = marvin_dap_maps.get_bpt(use_oi=False, return_figure=False, show_plot=False)

        bptclasses = [bptmap[c]['global'] for c in self.categories]
        # 'other' denotes where no other mask is set
        other = np.ones_like(bptclasses[0])
        bptclasses.append(other)
        # if any BPT category is set to TRUE, OTHER is ignored
        # b/c numpy chooses first instance of max value
        all_bptclasses = np.stack(bptclasses, axis=0)

        assigned_bptclass = np.argmax(all_bptclasses, axis=0)

        return assigned_bptclass

class Binner(object):
    '''
    bins data according to bin definitions
    '''
    def __init__(self, bindefs, *args, **kwargs):
        self.bindefs = bindefs
        self.shape = tuple(d.nbins for d in self.bindefs)
        

    @classmethod
    def from_namesbounds(cls, *args):
        '''
        construct from alternating names and bin-bounds arguments
        '''
        allnames, allbounds = args[::2], args[1::2]
        allbindefs = list(map(BinDef, allnames, allbounds, range(len(allnames))))
        return cls(allbindefs)

    def __call__(self, vals):
        if len(vals) != len(self.bindefs):
            raise ValueError(
                'shape mismatch: axis 0 of vals should have size equal to number of bin axes')
        return tuple(bindef(val) for bindef, val in zip(self.bindefs, vals))

    @lazyproperty
    def table(self):
        bin_llims = np.meshgrid(*[bd.binbounds[:-1] for bd in self.bindefs], indexing='ij')
        bin_ulims = np.meshgrid(*[bd.binbounds[1:] for bd in self.bindefs], indexing='ij')
        bin_ix = np.meshgrid(*list(range(s) for s in self.shape))

        tab = t.Table(data=[range(np.prod(self.shape))], names=['num'])
        llims_names = ['{}_llim'.format(bd.name) for bd in self.bindefs]
        ulims_names = ['{}_ulim'.format(bd.name) for bd in self.bindefs]
        ix_names = ['{}_ix'.format(bd.name) for bd in self.bindefs]

        for ll, lln, ul, uln, ix, ixn in zip(
            bin_llims, llims_names, bin_ulims, ulims_names, bin_ix, ix_names):

            tab[lln] = ll.flatten()
            tab[uln] = ul.flatten()
            tab[ixn] = ix.flatten()

        tab.add_index('num')

        return tab

    def binnum_to_binixs(self, num):
        return np.unravel_index(num, self.shape)

    def define_binstacks(self, lamgrid):
        self.bin_stacks = [BinStack(
                               lamgrid, bin_ix=np.array(self.binnum_to_binixs(binnum)),
                               binnum=binnum)
                           for binnum in self.table['num']]


def write_bin_assign(bin_assignment, plateifu, loc='.'):
    '''write bin assignment to FITS

    Parameters
    ----------
    bin_assignment : tuple of np.ma.array
        tuple of numpy mask arrays, giving bin assignment along each bin axis

    plateifu : str
        plateifu designation

    loc : str
        where output file gets placed
    '''

    hdulist = fits.HDUList(fits.PrimaryHDU())
    bin_a = np.ma.stack(bin_assignment, axis=0)
    
    bin_assign_hdu = fits.ImageHDU(bin_a.data)
    bin_assign_hdu.header['EXTNAME'] = 'ASSIGN'
    hdulist.append(bin_assign_hdu)

    mask_hdu = fits.ImageHDU(bin_a.mask.any(axis=0).astype(float))
    mask_hdu.header['EXTNAME'] = 'MASK'
    hdulist.append(mask_hdu)

    hdulist.writeto(os.path.join(loc, '{}_BINASSIGN.fits'.format(plateifu)), overwrite=True)