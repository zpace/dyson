import numpy as np
from astropy import table as t

import bpt

import utils as ut

class BinDef(object):
    '''
    defines a bin
    '''
    def __init__(self, name, binbounds, index, *args, 
                 fn=None, fn_args=tuple(), fn_kwargs=dict(), **kwargs):
        self.name = name
        self.binbounds = binbounds
        self.binbounds_ext = np.concatenate([[-np.inf], self.binbounds, [np.inf]])
        self.nbins = len(self.binbounds) + 1
        self.index = index

        if fn is None:
            self.fn = lambda x, *args, **kwargs: x

        self.fn_args, self.fn_kwargs = fn_args, fn_kwargs

    def __call__(self, vals):
        return np.digitize(self.fn(vals, *self.fn_args, **self.fn_kwargs), bins=self.binbounds)

    @property
    def table(self):
        index_col_name = '{}_index'.format(self.name)
        tab = t.Table(data=[self.binbounds_ext[:-1], self.binbounds_ext[1:], range(self.nbins)],
                      names=['{}_low'.format(self.name), '{}_high'.format(self.name), 
                             index_col_name])
        tab.add_index(index_col_name)
        return tab


class TableBinDef(BinDef):
    '''
    defines a bin which lives in a table
    '''
    def __init__(self, tab, ix_colname, qty_colname, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = dict(zip(tab[ix_colname], tab[qty_colname]))

    def __call__(self, ix):
        return np.digitize(self.fn(self.mapping[ix], *self.fn_args, **self.fn_kwargs), bins=self.binbounds)

class BPTBinDef(BinDef):
    '''
    defines a bin based on ionization diagnostic
    '''
    def __init__(self, dap_maps, *args, **kwargs):
       super().__init__(*args, **kwargs)

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
        allnames, albounds = args[::2], args[1::2]
        allbindefs = list(map(BinDef, allnames, allbounds, range(len(allnames))))
        return cls(allbindefs)

    def __call__(self, vals):
        if len(vals) != len(self.bindefs):
            raise ValueError(
                'shape mismatch: axis 0 of vals should have size equal to number of bin axes')
        return tuple(bindef(vals[bindef.index]) for bindef in self.bindefs)

    @property
    def table(self):
        tabs = {bindef.name: bindef.table for bindef in self.bindefs}