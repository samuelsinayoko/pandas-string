"""Minimal example showing how to consume pyarrow strings with numba.
"""
import math

import numba
import numpy as np
import pyarrow as pa


@numba.jitclass([
    ('start', numba.uint32),
    ('end', numba.uint32),
    ('data', numba.uint8[:]),
])
class NumbaString:
    """Numba wrapper for PyArrow strings.

    Example
    -------
    >>> mystring = "
    >>> NumbaStringArray.make(array)

    """

    def __init__(self, data, start=0, end=None):
        if end is None:
            end = data.shape[0]

        self.data = data
        self.start = start
        self.end = end

    @property
    def length(self):
        return self.end - self.start

    def get_byte(self, i):
        return self.data[self.start + i]


def _make_string(obj):
    if isinstance(obj, str):
        data = obj.encode('utf8')
        data = np.asarray(memoryview(data))

        return NumbaString(data, 0, len(data))

    raise TypeError()


NumbaString.make = _make_string
