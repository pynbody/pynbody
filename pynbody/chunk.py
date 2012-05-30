from __future__ import division
import random
import math
import numpy as np


print "I am chunk.py.  I have not single a comment........  Why am I here?"

class Chunk:
    def __init__(self, *args, **kwargs):
        pass
        start,stop,step = 0, None, 1
        if len(args) == 1: start,stop,step = 0, args[0], 1
        if len(args) == 2: start,stop,step = args[0], args[1], 1
        if len(args) == 3: start,stop,step = args
        self.random = kwargs.get('random', None)
        self.ids    = kwargs.get('ids', None)

        self.start = start
        self.stop  = stop
        self.step  = step

        assert (self.step is not None 
            or  self.random is not None
            or  self.ids is not None)


    def init(self, max_stop):

        assert max_stop >= 0

        if self.stop is None:
            self.stop = max_stop
        else:
            self.stop = min(max_stop, self.stop)

        assert self.start >= 0
        assert self.stop >= 0, '%ld' % self.stop
        assert self.step > 0
        assert self.start <= self.stop

        if self.random is not None:
            assert random > 0

        if self.ids is not None:
            assert np.amax(self.ids) < max_stop
            if self.random is not None:
                if self.random < 1: self.random = int(self.random * len(self.ids))
                self.ids = random.sample(self.ids, self.random)
                self.ids.sort()
            self.len = len(self.ids)
        else:
            slice_len = int(math.ceil((self.stop-self.start) / self.step))
            self.len = slice_len
            if self.random is not None:
                if self.random < 1: self.random = int(self.random * self.len)
                self.random = min(self.random, slice_len)
                self.ids = random.sample(xrange(self.start, self.stop, self.step), self.random)
                self.ids.sort()
                self.len = len(self.ids)

    def __len__(self):
        return self.len

    def pdeltas(self, step=None):

        assert step is None or self.contiguous()
        if step is None:
            step = self.step

        if self.ids is None:
            for i in xrange(self.start, self.stop, self.step):
                yield self.step
        else:
            yield self.ids[0]
            for i in xrange(len(self.ids)-1):
                yield self.ids[i+1] - self.ids[i]


    def contiguous(self):
        return self.ids is None and self.step == 1
