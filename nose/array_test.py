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


    if hasattr(np.mean(x),'units') :
        assert np.var(x).units=="kpc**2"
        assert np.std(x).units=="kpc"
        assert np.mean(x).units=="kpc"

def test_iop_units() :
    x= SA([1,2,3,4])
    x.units = 'kpc'

    y = SA([2,3,4,5])
    y.units = 'km s^-1'

    z = SA([1000,2000,3000,4000])
    z.units = 'm s^-1'

    print(repr(x))
    
    try :
        x+=y
        assert False # above operation is invalid
    except ValueError :
        pass

  
    x*=pynbody.units.Unit('K')
    
    assert x.units=='K kpc'

    x.units = 'kpc'

    x*=y

    assert x.units=='km s^-1 kpc'
    assert (x==[2,6,12,20]).all()

    y+=z
    assert y.units=='km s^-1'

    assert (y==[3,5,7,9]).all()
    


def test_iop_sanity() :
    x = SA([1,2,3,4])
    x_id = id(x)
    x+=1
    assert id(x)==x_id
    x-=1
    assert id(x)==x_id
    x*=2
    assert id(x)==x_id
    x/=2
    assert id(x)==x_id
    

def test_unit_array_interaction() :
    """Test for issue 113 and related"""
    x  = pynbody.units.Unit('1 Mpc')
    y = SA(np.ones(10),'kpc')
    assert all(x+y == SA([ 1.001] * 10, 'Mpc'))
    assert all(x-y == SA([ 0.999] * 10, 'Mpc'))
    assert (x+y).units=='Mpc'
    assert all(y+x == SA([ 1.001] * 10, 'Mpc'))
    assert all(y-x == SA([ -0.999] * 10, 'Mpc'))
