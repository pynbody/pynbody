from .analysis import halo

class translate(object) :
    def __init__(self, f, shift) :
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
        """

        
        self.f = f
        self.shift = shift

    def __enter__(self) :
        self.f['pos']+=self.shift

    def __exit__(self, *args) :
        self.f['pos']-=self.shift

class v_translate(object) :
    def __init__(self, f, shift) :
        """Form a context manager for translating the simulation *f* by the given
        velocity *shift*.

        This is equivalent to the translate context manager, only operating on velocities instead
        of positions."""
        self.f = f
        self.shift = shift

    def __enter__(self) :
        self.f['vel']+=self.shift

    def __exit__(self, *args) :
        self.f['vel']-=self.shift

class xv_translate(object) :
    def __init__(self, f, xshift, vshift) :
        """Form a context manager for translating the simulation *f* by the given
        position *xshift* and velocity *vshift*.

        This is equivalent to the translate context manager, only operating on velocities 
        as well as positions."""
        self.f = f
        self.xshift = xshift
        self.vshift = vshift

    def __enter__(self) :
        self.f['pos']+=self.xshift
        self.f['vel']+=self.vshift

    def __exit__(self, *args) :
        self.f['pos']-=self.xshift
        self.f['vel']-=self.vshift

class center(object) :
    def __init__(self, h, mode=None, vel=True, recenter_ancestor=True) :
        self.h = h
        self.mode = mode
        self.vel = vel
        if recenter_ancestor :
            self.f = h.ancestor
        else :
            self.f = h
            
        self.cen = None
        self.vcen=None

    def __enter__(self) :
        if self.cen is None :
            self.cen = halo.center(self.h, mode=self.mode, retcen=True)
        self.f['pos']-=self.cen
        if self.vel :
            if self.vcen is None :
                try:
                    self.vcen = halo.vel_center(self.h, retcen=True)
                except:
                    self.f['pos']+=self.cen
                    raise
            self.f['vel']-=self.vcen

    def __exit__(self,*args) :
        self.f['pos']+=self.cen
        if self.vel :
            self.f['vel']+=self.vcen
