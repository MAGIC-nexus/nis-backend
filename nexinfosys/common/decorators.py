import functools
import warnings

# Decorators from "https://wiki.python.org/moin/PythonDecoratorLibrary"


def singleton(cls):
    """ Use class as singleton. 

 Sample use:


@singleton
class Foo:
    def __new__(cls):
        cls.x = 10
        return object.__new__(cls)

    def __init__(self):
        assert self.x == 10
        self.x = 15

assert Foo().x == 15
Foo().x = 20
assert Foo().x == 20    
    
    """

    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kw):
        it =  cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__

    return cls


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    
=== Examples of use ===

@deprecated
def some_old_function(x,y):
    return x + y

class SomeClass:
    @deprecated
    def some_old_method(self, x,y):
        return x + y
    
    """
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


class countcalls(object):
    """ Decorator that keeps track of the number of times a function is called. """

    __instances = {}

    def __init__(self, f):
        self.__f = f
        self.__numcalls = 0
        countcalls.__instances[f] = self

    def __call__(self, *args, **kwargs):
        self.__numcalls += 1
        return self.__f(*args, **kwargs)

    @staticmethod
    def count(f):
        """ Return the number of times the function f was called """
        return countcalls.__instances[f].__numcalls

    @staticmethod
    def counts():
        """ Return a dict of {function: # of calls} for all registered functions """
        return dict([(f, countcalls.count(f)) for f in countcalls.__instances])
