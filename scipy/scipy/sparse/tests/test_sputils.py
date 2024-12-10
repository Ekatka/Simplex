"""unit tests for sparse utility functions"""

import numpy as np
from numpy.testing import assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.sparse import _sputils as sputils
from scipy.sparse._sputils import matrix


class TestSparseUtils:

    def test_upcast(self):
        assert_equal(sputils.upcast('intc'), np.intc)
        assert_equal(sputils.upcast('int32', 'float32'), np.float64)
        assert_equal(sputils.upcast('bool', complex, float), np.complex128)
        assert_equal(sputils.upcast('i', 'd'), np.float64)

    def test_getdtype(self):
        A = np.array([1], dtype='int8')

        assert_equal(sputils.getdtype(None, default=float), float)
        assert_equal(sputils.getdtype(None, a=A), np.int8)

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype object. .*",
        ):
            sputils.getdtype("O")

        with assert_raises(
            ValueError,
            match="scipy.sparse does not support dtype float16. .*",
        ):
            sputils.getdtype(None, default=np.float16)

    def test_isscalarlike(self):
        assert_equal(sputils.isscalarlike(3.0), True)
        assert_equal(sputils.isscalarlike(-4), True)
        assert_equal(sputils.isscalarlike(2.5), True)
        assert_equal(sputils.isscalarlike(1 + 3j), True)
        assert_equal(sputils.isscalarlike(np.array(3)), True)
        assert_equal(sputils.isscalarlike("16"), True)

        assert_equal(sputils.isscalarlike(np.array([3])), False)
        assert_equal(sputils.isscalarlike([[3]]), False)
        assert_equal(sputils.isscalarlike((1,)), False)
        assert_equal(sputils.isscalarlike((1, 2)), False)

    def test_isintlike(self):
        assert_equal(sputils.isintlike(-4), True)
        assert_equal(sputils.isintlike(np.array(3)), True)
        assert_equal(sputils.isintlike(np.array([3])), False)
        with assert_raises(
            ValueError,
            match="Inexact indices into sparse matrices are not allowed"
        ):
            sputils.isintlike(3.0)

        assert_equal(sputils.isintlike(2.5), False)
        assert_equal(sputils.isintlike(1 + 3j), False)
        assert_equal(sputils.isintlike((1,)), False)
        assert_equal(sputils.isintlike((1, 2)), False)

    def test_isshape(self):
        assert_equal(sputils.isshape((1, 2)), True)
        assert_equal(sputils.isshape((5, 2)), True)

        assert_equal(sputils.isshape((1.5, 2)), False)
        assert_equal(sputils.isshape((2, 2, 2)), False)
        assert_equal(sputils.isshape(([2], 2)), False)
        assert_equal(sputils.isshape((-1, 2), nonneg=False),True)
        assert_equal(sputils.isshape((2, -1), nonneg=False),True)
        assert_equal(sputils.isshape((-1, 2), nonneg=True),False)
        assert_equal(sputils.isshape((2, -1), nonneg=True),False)

        assert_equal(sputils.isshape((1.5, 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape(([2], 2), allow_nd=(1, 2)), False)
        assert_equal(sputils.isshape((2, 2, -2), nonneg=True, allow_nd=(1, 2)),
                     False)
        assert_equal(sputils.isshape((2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2,), allow_nd=(1, 2)), True)
        assert_equal(sputils.isshape((2, 2, 2), allow_nd=(1, 2)), False)

    def test_issequence(self):
        assert_equal(sputils.issequence((1,)), True)
        assert_equal(sputils.issequence((1, 2, 3)), True)
        assert_equal(sputils.issequence([1]), True)
        assert_equal(sputils.issequence([1, 2, 3]), True)
        assert_equal(sputils.issequence(np.array([1, 2, 3])), True)

        assert_equal(sputils.issequence(np.array([[1], [2], [3]])), False)
        assert_equal(sputils.issequence(3), False)

    def test_ismatrix(self):
        assert_equal(sputils.ismatrix(((),)), True)
        assert_equal(sputils.ismatrix([[1], [2]]), True)
        assert_equal(sputils.ismatrix(np.arange(3)[None]), True)

        assert_equal(sputils.ismatrix([1, 2]), False)
        assert_equal(sputils.ismatrix(np.arange(3)), False)
        assert_equal(sputils.ismatrix([[[1]]]), False)
        assert_equal(sputils.ismatrix(3), False)

    def test_isdense(self):
        assert_equal(sputils.isdense(np.array([1])), True)
        assert_equal(sputils.isdense(matrix([1])), True)

    def test_validateaxis(self):
        assert_raises(TypeError, sputils.validateaxis, (0, 1))
        assert_raises(TypeError, sputils.validateaxis, 1.5)
        assert_raises(ValueError, sputils.validateaxis, 3)

        # These function calls should not raise errors
        for axis in (-2, -1, 0, 1, None):
            sputils.validateaxis(axis)

    def test_get_index_dtype(self):
        imax = np.int64(np.iinfo(np.int32).max)
        too_big = imax + 1

        # Check that uint32's with no values too large doesn't return
        # int64
        a1 = np.ones(90, dtype='uint32')
        a2 = np.ones(90, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # Check that if we can not convert but all values are less than or
        # equal to max that we can just convert to int32
        a1[-1] = imax
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int32')
        )

        # Check that if it can not convert directly and the contents are
        # too large that we return int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), check_contents=True)),
            np.dtype('int64')
        )

        # test that if can not convert and didn't specify to check_contents
        # we return int64
        a1 = np.ones(89, dtype='uint32')
        a2 = np.ones(89, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2))),
            np.dtype('int64')
        )

        # Check that even if we have arrays that can be converted directly
        # that if we specify a maxval directly it takes precedence
        a1 = np.ones(12, dtype='uint32')
        a2 = np.ones(12, dtype='uint32')
        assert_equal(
            np.dtype(sputils.get_index_dtype(
                (a1, a2), maxval=too_big, check_contents=True
            )),
            np.dtype('int64')
        )

        # Check that an array with a too max size and maxval set
        # still returns int64
        a1[-1] = too_big
        assert_equal(
            np.dtype(sputils.get_index_dtype((a1, a2), maxval=too_big)),
            np.dtype('int64')
        )

    # tests public broadcast_shapes largely from
    # numpy/numpy/lib/tests/test_stride_tricks.py
    # first 3 cause np.broadcast to raise index too large, but not sputils
    @pytest.mark.parametrize("input_shapes,target_shape", [
        [((6, 5, 1, 4, 1, 1), (1, 2**32), (2**32, 1)), (6, 5, 1, 4, 2**32, 2**32)],
        [((6, 5, 1, 4, 1, 1), (1, 2**32)), (6, 5, 1, 4, 1, 2**32)],
        [((1, 2**32), (2**32, 1)), (2**32, 2**32)],
        [[2, 2, 2], (2,)],
        [[], ()],
        [[()], ()],
        [[(7,)], (7,)],
        [[(1, 2), (2,)], (1, 2)],
        [[(2,), (1, 2)], (1, 2)],
        [[(1, 1)], (1, 1)],
        [[(1, 1), (3, 4)], (3, 4)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[(5, 6, 1)], (5, 6, 1)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
        [[(1,), (3,)], (3,)],
        [[2, (3, 2)], (3, 2)],
        [[(1, 2)] * 32, (1, 2)],
        [[(1, 2)] * 100, (1, 2)],
        [[(2,)] * 32, (2,)],
    ])
    def test_broadcast_shapes_successes(self, input_shapes, target_shape):
        assert_equal(sputils.broadcast_shapes(*input_shapes), target_shape)

    # tests public broadcast_shapes failures
    @pytest.mark.parametrize("input_shapes", [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [2, (2, 3)],
        [(3,), (3,), (4,)],
        [(2, 5), (3, 5)],
        [(2, 4), (2, 5)],
        [(1, 3, 4), (2, 3, 3)],
        [(1, 2), (3, 1), (3, 2), (10, 5)],
        [(2,)] * 32 + [(3,)] * 32,
    ])
    def test_broadcast_shapes_failures(self, input_shapes):
        with assert_raises(ValueError, match="cannot be broadcast"):
            sputils.broadcast_shapes(*input_shapes)

    def test_check_shape_overflow(self):
        new_shape = sputils.check_shape([(10, -1)], (65535, 131070))
        assert_equal(new_shape, (10, 858967245))

    def test_matrix(self):
        a = [[1, 2, 3]]
        b = np.array(a)

        assert isinstance(sputils.matrix(a), np.matrix)
        assert isinstance(sputils.matrix(b), np.matrix)

        c = sputils.matrix(b)
        c[:, :] = 123
        assert_equal(b, a)

        c = sputils.matrix(b, copy=False)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])

    def test_asmatrix(self):
        a = [[1, 2, 3]]
        b = np.array(a)

        assert isinstance(sputils.asmatrix(a), np.matrix)
        assert isinstance(sputils.asmatrix(b), np.matrix)

        c = sputils.asmatrix(b)
        c[:, :] = 123
        assert_equal(b, [[123, 123, 123]])