import pynbody
import numpy as np

def test_order_bridge() :
    f1 = pynbody.new(dm=1000)
    f2 = pynbody.new(dm=3000)
    f1['iord'] = np.arange(5,2005,2,dtype=int)
    f2['iord'] = np.arange(3000,dtype=int)
    
    b = pynbody.bridge.OrderBridge(f1,f2)

    h1 = f1[:50:2]
    assert b(h1).ancestor is f2
    assert (b(h1)['iord']==h1['iord']).all
