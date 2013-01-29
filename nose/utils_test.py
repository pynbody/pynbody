import pynbody, numpy as np

def random_slice(max_pos=1000, max_step=10) :
    import random
    a = random.randint(0,max_pos)
    b = random.randint(0,max_pos)
    return slice(min(a,b),max(a,b),random.randint(1,max_step))

def test_intersect_slices() :
    """Unit test for intersect_slices, relative_slice and chained_slice"""
    from pynbody.util import intersect_slices

    numbers = np.arange(1000)

    for x in xrange(1000) :
	s1= random_slice()
	s2 = random_slice()
	n1 = numbers[s1]
	n2 = numbers[s2]
	s_inter = pynbody.util.intersect_slices(s1,s2,len(numbers))
	nx = numbers[s_inter]
	nx2 = numbers[s1][pynbody.util.relative_slice(s1,s_inter)]
	correct_intersection=set(n1).intersection(set(n2))
	assert set(nx)==correct_intersection
	assert set(nx2)==correct_intersection

	s1 = random_slice()
	s2 = random_slice(20,4)
	n3 = numbers[s1][s2]
	s3 = pynbody.util.chained_slice(s1,s2)
	nx3 = numbers[s3]
	assert len(n3)==len(nx3)
	assert all([x==y for x,y in zip(n3,nx3)])

	rel_slice_to_self = pynbody.util.relative_slice(s1,s1)
	assert rel_slice_to_self.step is None
	assert rel_slice_to_self.start==0

    # test 'None' steps handled correctly
    assert pynbody.util.intersect_slices(slice(0,5,None),slice(0,5,None))==slice(0,5,None)
    

def test_intersect_indices() :
    """Unit test for index_before_slice"""

    numbers = np.arange(1000)
    
    for x in xrange(1000) :
	sl = random_slice()
	t = numbers[sl]
	if len(t)==0 :
	    continue
	ind = np.random.randint(0,len(t),50)
	new_ind = pynbody.util.index_before_slice(sl, ind)
    
	assert len(numbers[new_ind])==len(numbers[sl][ind])
	
	assert all([x==y for x, y in zip(numbers[new_ind],numbers[sl][ind])])


def test_bisect() :
    """Unit test for bisection algorithm"""

    assert abs(pynbody.util.bisect(0.,1., lambda x: x**2-0.04)-0.20) < 1.e-6

def test_slice_length() :
    N = np.arange(100)

    for end in xrange(100) :
        for start in xrange(end) :
            for step in xrange(1,10) :
                S = slice(0,end,step)
                assert len(N[S]) == pynbody.util.indexing_length(S)
