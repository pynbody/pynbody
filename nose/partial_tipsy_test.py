import pynbody
import numpy as np

def test_indexing() :
    f1 = pynbody.load("testdata/g15784.lr.01024")

    test_len = 12310
    for test_len in [100, 10000, 20000] :
        for i in range(5) :
            subindex = np.random.permutation(np.arange(0,len(f1)))[:test_len]
            subindex.sort()
            f2 = pynbody.load("testdata/g15784.lr.01024", take=subindex)


            assert (f2['x']==f1[subindex]['x']).all()    
            assert (f2['iord']==f1[subindex]['iord']).all()

    
