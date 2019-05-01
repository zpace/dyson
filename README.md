# dyson

`dyson` is a package which takes spectra with similar inferred properties, and stacks their fitting residuals together. Spectra can be binned in arbitrary dimension.

The overall workflow is as follows: 

1) define binning space, consisting of many hyperrectangular bins `B`
2) define a wavelength grid `L`, having an identical log-wavelength spacing to the data
3) For each galaxy `G` in a list, assign each spaxel `S` within to a bin `B`, and write out the assignment map to a file `PLATEIFU-assign.fits`
4) for each bin `B`, loop through list of galaxies `G`, and find spaxels `S` in `G` which fit in `B`. For each spaxel `S`:
	a) find the observed-frame stellar-continuum-fit residual belonging to `S`. Deredshift to rest-frame, and drizzle into `L` (along with weight vector)
	b) add rest-frame, rectified residual (and weight) to running total for bin
	c) add spaxel redshifts to running list
	d) write running flux sum, variance, and redshift list to a file `BINNUM-accum.fits`
