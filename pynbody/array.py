import numpy as np

class SimArray(np.ndarray) :
    def __new__(subtype, data, units=None, **kwargs) :
	new = np.array(data, **kwargs).view(subtype)
	new.units = units
	return new

    @property
    def units(self) :
	raise RuntimeError("not implemented")

    @units.setter
    def units(self, value=None) :
	pass
