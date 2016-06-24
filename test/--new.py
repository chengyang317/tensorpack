


class Cheese(object):
    def __new__(cls, *args, **kwargs):

        obj = super(Cheese, cls).__new__(cls)
        num_holes = kwargs.get('num_holes', 0)

        if num_holes == 0:
            cls.__init__ = cls.foomethod
        else:
            cls.__init__ = cls.barmethod

        return obj

    def foomethod(self, *args, **kwargs):
        print "foomethod called as __init__ for Cheese"

    def barmethod(self, *args, **kwargs):
        print "barmethod called as __init__ for Cheese"


class son(Cheese):
    def __init__(self, *args, **kwargs):
        print 'hello'

if __name__ == "__main__":
    parm = Cheese(num_holes=5)
    a = son(num_holes=5)
    pass