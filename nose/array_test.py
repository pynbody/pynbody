import pynbody
SA = pynbody.array.SimArray
import numpy as np

def test_pickle() :
    import pickle
    x = SA([1,2,3,4],units='kpc')
    assert str(x.units)=='kpc'
    y = pickle.loads(pickle.dumps(x))
    assert y[3]==4
    assert str(y.units)=='kpc'
    
def test_return_types() :
    

    x = SA([1,2,3,4])
    y = SA([2,3,4,5])

    assert type(x) is SA
    assert type(x**2) is SA
    assert type(x+y) is SA
    assert type(x*y) is SA
    assert type(x**y) is SA
    assert type(2**x) is SA
    assert type(x+2) is SA
    assert type(x[:2]) is SA
    

    x2d = SA([[1,2,3,4],[5,6,7,8]])
    
    assert type(x2d.sum(axis=1)) is SA

def test_unit_tracking() :
    
    x = SA([1,2,3,4])
    x.units = "kpc"

    y = SA([5,6,7,8])
    y.units = "Mpc"

    assert abs((x*y).units.ratio("kpc Mpc")-1.0)<1.e-9

    assert ((x**2).units.ratio("kpc**2")-1.0)<1.e-9

    assert ((x/y).units.ratio("")-1.e-3)<1.e-12

    assert np.var(x).units=="kpc**2"
    
    assert np.std(x).units=="kpc"

    if np.__version__[0]>'1' :
        assert np.mean(x).units=="kpc"
