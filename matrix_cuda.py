from matrix import Matrix
import theano.tensor as T
from theano import config
from theano import function
import numpy as np

class MatrixCuda(Matrix):
    class TheanoFunctions(object):
        x = T.dmatrix('x')
        y = T.dmatrix('y')
        add = function([x, y], x + y)
        sub = function([x, y], x - y)
        dot = function([x, y], T.dot(x, y))
        inv = function([x], T.inv(x))


    def __init__(self, value=None):
        """ Create a matrix from list of lists. """
        self.value = np.asarray(np.array(value),config.floatX) if value is not None else np.asarray(np.array([[]]),config.floatX)
        self.dimx, self.dimy = self.value.shape
        self._T = None
        self._I = None

    def __add__(self, other):
        new = MatrixCuda.TheanoFunctions.add(self._value, other._value)
        return Matrix(new)

    def __sub__(self, other):
        new = MatrixCuda.TheanoFunctions.sub(self._value,other._value)
        return Matrix(new)

    def __mul__(self, other):
        new = MatrixCuda.TheanoFunctions.dot(self._value, other._value)
        return Matrix(new)

    def _inverse(self):
        new = MatrixCuda.TheanoFunctions.inv(self._value)
        return Matrix(new)
