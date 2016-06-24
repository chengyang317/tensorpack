# -*- coding: UTF-8 -*-
# File: inheritance
# Author: Philip Cheng
# Time: 6/24/16 -> 12:24 AM



class A(object):
    def __init__(self):
        pass

    def hello(self):
        print 'Hello from A'
        self._hi(message='nihao')

    def _hi(self, *args, **kwargs):
        print 'Hi from A'


class B(A):
    def __init__(self):
        pass

    def _hi(self, message):
        print 'message is {}'.format(message)

b = B()
b.hello()
pass
