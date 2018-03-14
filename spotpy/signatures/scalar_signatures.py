# -*- coding: utf-8 -*-
"""
Copyright 2017 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).
:author: Tobias Houska, Benjamin Manns, Philipp Kraft

This module calculates scalars from timeseries to describe signature behaviour.

The signature behaviour indices are collected from different sources:

.. [WESMCM2015] "Uncertainty in hydrological signatures"
            by I. K. Westerberg and H. K. McMillan, 2015
            `DOI: 10.5194/hess-19-3951-2015 <https://doi.org/10.5194/hess-19-3951-2015>`_
.. [CLBGS2000] "Flow variables for ecological studies in temperate streams: groupings based on covariance",
                B Clausen, BJF Biggs 2000
                `DOI:10.1016/S0022-1694(00)00306-1 <https://doi.org/10.1016/S0022-1694(00)00306-1>`_

All methods without further parameters in this module should follow
the following template, where "signature" is replaced with the actual name

>>> def get_signature(data, stepsize=None):
...     return sum(data)

**Where:**
- `data` is the data time series as any sequence of floats
- `stepsize` is the raster length of the timeseries in seconds.
  For daily time series, `stepsize=86400`. Many signatures ignore the `stepsize`parameter,
  but it needs to be provided to ensure a consistent interface

TODO: Check if `stepsize` can be removed from the interface

If a method needs additional parameters, create a class for that method,
with the parameters as attributes of an instance and a __call__ method
with the same interface as the function above.

>>> class Signature(__BaseSignature):
...     def __init__(self, parameter):
...         self.parameter = parameter
...     def __call__(self, data, stepsize=None):
...         return sum(data) ** self.parameter

For typical parametrizations, create instances of this class inside this module,
cf. to QuantileSignature and the get_qXXX methods
"""


import numpy as np
import sys

if sys.version_info[0] >= 3:
    from inspect import getdoc as _getdoc
    unicode = str
else:
    def _getdoc(obj):
        u = obj.__doc__
        try:
            return u'\n'.join(l.strip() for l in u.split(u'\n') if l.strip())
        except UnicodeDecodeError:
            raise AssertionError(
                '{}: Docstring uses unicode but {} misses the line ``from __future__ import unicode_literals``'
                .format(obj, type(obj).__module__)
                )


def remove_nan(data):
    """
    Returns the the data timeseries with NaN and inifinite values removed

    :param data: The timeseries data as a numeric sequence
    :return: data[np.isfinite(data)], might be shorter than the input
    """
    return np.array(data)[np.isfinite(data)]


def fill_nan(data):
    """
    Returns the timeseries where any gaps (represented by NaN) are
    filled, using a linear approximation between the neighbors

    :param data: The timeseries data as a numeric sequence
    :return:
    """

    # TODO: Write an implementation to fill NaN values
    raise NotImplementedError("Implementation details are missing")


class QuantileSignature(object):
    """
    Calculates the <quantile>% percentile from a runoff time series.

    Used as a signature behaviour index by [WESMCM2015]_
    """
    def __init__(self, quantile):
        self.percentile = quantile
        self.__doc__ = _getdoc(type(self)).replace('<quantile>', '{:0.4g}'.format(self.percentile))

    def __call__(self, data, stepsize=None):
        """
        Calculates the flow <quantile>% of the time exceeded from a runoff time series.

        :param data: The timeseries data as a numeric sequence
        :param stepsize: Unused
        :return: quantile signature behaviour index
        """
        return np.percentile(remove_nan(data), self.percentile)

    def __repr__(self):
        return 'q({:0.2f}%)'.format(self.percentile)


get_q0_01 = QuantileSignature(0.01)
get_q0_1 = QuantileSignature(0.1)
get_q1 = QuantileSignature(1)
get_q5 = QuantileSignature(5)
get_q50 = QuantileSignature(50)
get_q85 = QuantileSignature(85)
get_q95 = QuantileSignature(95)
get_q99 = QuantileSignature(99)


def get_mean(data, stepsize=None):
    """
    Calculates the mean from a runoff time series.

    Used as a signature behaviour index by [WESMCM2015]_ and [CLBGS2000]_

    :param data: The runoff timeseries data as a numeric sequence
    :param stepsize: the stepsize of the timeseries (unused)
    :return: A single number containing the signature behaviour index
    """
    return np.mean(remove_nan(data))


def get_skewness(data, stepsize=None):
    """
    Skewness, i.e. the mean flow data divided by Q50.

    See: [CLBGS2000]_ and [WESMCM2015]_

    :param data: The runoff timeseries data as a numeric sequence
    :param stepsize: the stepsize of the timeseries (unused)
    :return: A single number containing the signature behaviour index

    """
    return get_mean(data) / get_q50(data)


def get_cv(data, stepsize=None):
    """
    Coefficient of variation, i.e. standard deviation divided by mean flow

    See: [CLBGS2000]_

    :param data: The runoff timeseries data as a numeric sequence
    :param stepsize: the stepsize of the timeseries (unused)
    :return: A single number containing the signature behaviour index

    """
    return remove_nan(data).std() / get_mean(data)
