import numpy as np
from .analysis import halo
from . import snapshot

class Transformation(object) :
    def __init__(self, f, defer=False) :
        if not (isinstance(f, snapshot.SimSnap) or isinstance(f, Transformation)) :
            raise ValueError, "Transformation must either act on another Transformation or on a SimSnap"
        self.f = f
        self.applied = False
        if not defer :
            self.apply(force=False)

    def apply(self, force=True) :
        
        if isinstance(self.f, snapshot.SimSnap) :
            f = self.f
        else :
            # this is a chained transformation, get the SimSnap to operate on
            # from the level below
            f = self.f.apply(force=force)
            
        if self.applied and force :
            raise RuntimeError, "Transformation has already been applied"
       
        if not self.applied :
            self._apply(f)
            self.applied = True
            self._applied_to = f

        return f
        
    def revert(self) :
        if not self.applied :
            raise RuntimeError, "Transformation has not been applied"
        self._revert(self._applied_to)
        
        if isinstance(self.f, Transformation) :
            self.f.revert()
         

    def __enter__(self) :
        self.apply(force=False)

    def __exit__(self, *args) :
        self.revert()

        
class GenericTranslate(Transformation) :
    def __init__(self, f, arname, shift) :
        self.shift = shift
        self.arname = arname
        super(GenericTranslate, self).__init__(f)
        
    def _apply(self, f) :
        f[self.arname]+=self.shift

    def _revert(self, f) :
        f[self.arname]-=self.shift

class GenericRotation(Transformation) :
    def __init__(self, f, matrix, ortho_tol=1.e-8) :
        # Check that the matrix is orthogonal
        resid = np.dot(matrix, np.asarray(matrix).T) - np.eye(3)
        resid = (resid**2).sum()
        if resid > ortho_tol or resid != resid:
            raise ValueError("Transformation matrix is not orthogonal")
        self.matrix = matrix
        super(GenericRotation, self).__init__(f)

    def _apply(self, f) :
        f._transform(self.matrix)

    def _revert(self, f) :
        f._transform(self.matrix.T)


def translate(f, shift) :
    """Form a context manager for translating the simulation *f* by the given
    spatial *shift*.

    This allows you to enclose a code block within which the simulation is offset
    by the specified amount. On exiting the code block, you are guaranteed the
    simulation is put back to where it started, so
    
    with translate(f, shift) :
        print f['pos'][0]

    is equivalent to

    try:
        f['pos']+=shift
        print f['pos'][0]
    finally:
        f['pos']-=shift

    On the other hand,

    translate(f, shift)
    print f['pos'][0]

    Performs the translation but does not revert it at any point.
    """
    return GenericTranslate(f, 'pos', shift)

def v_translate(f, shift) :
    """Form a context manager for translating the simulation *f* by the given
    velocity *shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""
    
    return GenericTranslate(f, 'vel', shift)

def xv_translate(f, x_shift, v_shift) :
    """Form a context manager for translating the simulation *f* by the given
    position *x_shift* and velocity *v_shift*.

    For a fuller description, see *translate* (which applies to position transformations)."""
    
    return translate(v_translate(f,v_shift),
                     x_shift)


def transform(f, matrix) :
    return GenericRotation(f,matrix)
