import numpy as np
from astropy import table as t

import utils as ut

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
        self.binbounds = binbounds
        self.binbounds_ext = np.concatenate([[-np.inf], self.binbounds, [np.inf]])
        self.nbins = len(self.binbounds) + 1
        self.index = index

        if fn is None:
            self.fn = lambda x, *args, **kwargs: x

        self.fn_args, self.fn_kwargs = args, kwargs

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
        return np.digitize(self.fn(vals, *self.fn_args, **self.fn_kwargs), bins=self.binbounds)

    @property
    def table(self):
        '''table of bin boundaries
        
        Returns
        -------
        astropy.table.Table
            table of bin boundaries
        '''
        index_col_name = '{}_index'.format(self.name)
        tab = t.Table(data=[self.binbounds_ext[:-1], self.binbounds_ext[1:], range(self.nbins)],
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