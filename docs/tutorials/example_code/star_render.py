import matplotlib.pylab as plt

import pynbody

# load the snapshot and set to physical units
s = pynbody.load('testdata/gasoline_ahf/g15784.lr.01024.gz')
s.physical_units()

# load the halos
halos = s.halos()

# center on the largest halo and align the disk
pynbody.analysis.angmom.sideon(halos[1])

#create an image using the default bands (i, v, u)
pynbody.plot.stars.render(s,width='20 kpc')
