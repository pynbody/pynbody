import sys, __builtin__


if sys.version_info[:2] <= (2, 5) :

    from .bc_modules import fractions

    # emulation of python 2.6 property class from
    # http://blog.yjl.im/2009/02/propery-setter-and-deleter-in-python-25.html

    class property(__builtin__.property):

        def __init__(self, fget, *args, **kwargs):

            self.__doc__ = fget.__doc__
            super(property, self).__init__(fget, *args, **kwargs)

        def setter(self, fset):

            cls_ns = sys._getframe(1).f_locals
            for k, v in cls_ns.iteritems():
                if v == self:
                    propname = k
                    break
            cls_ns[propname] = property(self.fget, fset,
                                        self.fdel, self.__doc__)
            return cls_ns[propname]

        def deleter(self, fdel):

            cls_ns = sys._getframe(1).f_locals
            for k, v in cls_ns.iteritems():
                if v == self:
                    propname = k
                    break
            cls_ns[propname] = property(self.fget, self.fset,
                                        fdel, self.__doc__)
            return cls_ns[propname]



else :
    property = __builtin__.property
    import fractions


if sys.version_info[:2] <= (2,6) :
    from .bc_modules import ordered_dict
    from bc_modules.ordered_dict import OrderedDict
else :
    import collections
    from collections import OrderedDict
    
