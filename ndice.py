import numpy as np
import utils as ut

class DiceNStack(object):
    '''
    data structure which holds spectra that are binned according to their properties
    '''
    def __init__(self, binner, nspec, cube_specax=0):
        self.binner = binner
        # set up denstination array
        self.stacks = np.zeros((nspec, ) + self.binner.shape)

        # for the arrays that we're targeting, what is the spectral axis?
        # assumption is that all remaining axes are ok to be sliced and diced apart
        self.cube_specax = cube_specax

    def process(self, target, resid):
        '''
        add residuals belonging to a galaxy with the appropriate bins (defined by `target`)
        '''
        map_shape = tuple(l for i, l in enumerate(resid.shape) if i != self.cube_specax)

        # make sure all bin assignments are arrays with the right shape
        bin_assgn = [b if b.type is np.ndarray else np.full(shape=map_shape, fill_value=b)
                     for b in self.binner(target)]
        dest = np.stack(bin_assgn, axis=0)

        # loop through spaxels
        for i, j in np.ndindex(map_shape):
            # make slices
            cube_sl = ut.mapslice_to_ndslice(i, j, new_ax=self.cube_specax, new_slice=slice(None))
            stacks_sl = ut.mapslice_to_ndslice(*dest[..., i, j], new_ax=0, new_slice=slice(None))

            self.stacks[stacks_sl] += resid[cube_sl]

