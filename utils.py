import numpy as np

import os

def dust_multicomponent(tau5500, lams_aa, diffuse_index=-0.7, dense_index=-1.3,
                        return_correct=False):
    '''
    Charlot and Fall two-component dust model
    '''
    taudiffuse_at_lams = tau5500 * (lams_aa / 5500.)**diffuse_index
    taudense_at_lams = tau5500 * (lams_aa / 5500.)**dense_index
    atten_factor = np.exp(-(taudiffuse_at_lams + taudense_at_lams))

    if return_correct:
        return 1. / atten_factor
    else:
        return atten_factor


def mapslice_to_ndslice(*ij, new_ax, new_slice):
    sl = list(ij)
    sl.insert(new_ax, new_slice)
    sl = tuple(sl)
    return sl