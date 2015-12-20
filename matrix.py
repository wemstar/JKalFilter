"""
A linear algebra module implementing the :py:class:`Matrix` class.
Based on:
https://github.com/ozzloy/udacity-cs373/blob/master/unit-2.py
"""
# pylint: disable=W0141,C0103
import numpy as np

class Matrix(object):

    """
    A matrix class covering all major matrix operations. Currently the
    operators +, - and * are supported. Matrix elements can also be accessed
    and modified using indexing.

    A matrix is constructed in one of two ways: passing a list of lists to the
    constructor or creating an :py:meth:`identity`/:py:meth:`zero` matrix and
    modifying the elements. If the constructor is invoked without parameters an empty matrix will be initialized from the list ``[[]]``.

        >>> A = Matrix([[1,2,3],[4,5,6]])
        >>> A
        Matrix([[  1.00,  2.00,  3.00],
                [  4.00,  5.00,  6.00]])
        >>> A[1][1]
        5
        >>> A[1][1] = 0.0
        >>> A
        Matrix([[  1.00,  2.00,  3.00],
                [  4.00,  0.00,  6.00]])

    Once a row has been accessed, the columns of that row can be sliced and
    modified.

        >>> A[1][:] = [5, 5, 5]
        Matrix([[  1.00,  2.00,  3.00],
                [  5.00,  5.00,  5.00]])

    By accessing the :py:attr:`I` or :py:attr:`T` fields the
    inverse/transposed matrix can be retrieved accordingly. An existing matrix
    may be modified by assigning a list of lists to the :py:attr:`value` field.

        >>> A.value = [[1, 1],[-1, 1]]
        >>> A
        Matrix([[  1.00,  1.00],
                [ -1.00,  1.00]])
        >>> A * A.I
        Matrix([[  1.00,  0.00],
                [  0.00,  1.00]])

    """

    def __init__(self, value=None):
        """ Create a matrix from list of lists. """
        self.value = np.array(value) if value is not None else np.array([[]])
        self.dimx, self.dimy = self.value.shape
        self._T = None
        self._I = None

    ################# Properties available externally ##################
    @property
    def value(self):
        """ Access the underlying data representation of the matrix, that is a
        list of list.

        :getter: get value
        :setter: set value"""
        # Can't be sure if the user uses this to change a matrix entry so have
        # to reset existing cached inverse and transpose.

        self._T = None
        self._I = None
        return self._value

    @value.setter
    def value(self, value):
        """ Set matrix value. """
        self._value = np.array(value)
        self.dimx, self.dimy = self._value.shape
        self._T = None
        self._I = None

    @property
    def T(self):
        """
        Get transposed matrix. The result is cached so that subsequent
        accesses to the transposed matrix are instantaneous. The result is
        uncached if the user accesses the matrix :py:attr:`.value` or a matrix
        element through indexing.
        """
        if self._T is None:
            self._T = self._transpose()
        return self._T

    ##################### Spawning special matrices ####################
    @classmethod
    def zero(cls, dimx, dimy):
        value = np.zeros((dimx, dimy))
        return cls(value)

    @classmethod
    def identity(cls, dim):
        """ Return an identity matrix of size *dim* x *dim*. """
        value = np.identity(dim)
        self = cls(value)
        return self

    ####################### Helper methods  ############################
    def show(self):
        """ Print the matrix. """
        for i in range(self.dimx):
            print self._value[i]
        print ' '

    def __repr__(self):
        """ Return string representation of matrix. """
        name = self.__class__.__name__ + "(["
        # class name and left bracket
        pad = len(name)
        join_string = ',\n' + ' ' * pad

        # formatting function
        def format_row(row):
            row = ["{:6.2f}".format(i) for i in row]
            row = ','.join(row)
            return '[' + row + ']'

        return name + join_string.join(map(format_row, self._value)) + "])"

    def __str__(self):
        """ Return the printed version of matrix. """
        return '[' + ",\n ".join(map(str, self._value)) + ']'

    def size(self):
        """
        Return the dimensions of the matrix.

        :returns: dimensions of the matrix *dimx* and *dimy*
        :rtype: *tuple*
        """
        return (self.dimx, self.dimy)

    def __getitem__(self, k):
        """ Return row of matrix. """
        return self.value[k]

    def _transpose(self):
        """ Return a transpose of the matrix. """
        value = self._value.T
        return Matrix(value)

    ############################ Arithmetics ###########################
    def __eq__(self, other):
        return (self.value == other.value).all()

    def __neq__(self, other):
        return (not self==other)

    def __add__(self, other):
        new = self._value + other._value
        return Matrix(new)

    def __sub__(self, other):
        new = self._value - other._value
        return Matrix(new)

    def __mul__(self, other):
        new = np.dot(self._value, other._value)
        return Matrix(new)

    def LU(self):
        """
        Return the LU decomposition of matrix, that is matrices :math:`L` and
        :math:`U` such that :math:`LU = \\text{self}`. Uses the Crout
        decomposition method, described at
        http://en.wikipedia.org/wiki/Crout_matrix_decomposition

        The input matrix needs to be square and the decomposition is actually
        performed on the pivoted matrix :math:`P \\cdot self` where
        :math:`P = self.pivot()`. The pivoting matrix is included as the first
        element of the return tuple.

        :return: matrices **P, L, U**
        :rtype: *tuple(Matrix)*
        """
        from scipy.linalg import lu
        P, L, U = map(Matrix, lu(self._value,permute_l=False))
        return (P, L, U)

    def _inverse(self):
        from numpy.linalg import inv
        new = Matrix(inv(self._value))
        return new

    @property
    def I(self):
        """
        Get inverse matrix through LU decomposition. The result is cached so
        that subsequent accesses to the inverse matrix are instantaneous. The
        result is uncached if the user accesses the matrix :py:attr:`.value`
        or a matrix element through indexing. This is equivalent to::

            decomposition = self.LU()
            return Matrix.LUInvert(*decomposition)
        """
        if self._I is None:
            self._I = self._inverse()
        return self._I
