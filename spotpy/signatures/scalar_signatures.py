# -*- coding: utf-8 -*-
"""
Copyright 2017 by Tobias Houska
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Tobias Houska, Benjamin Manns, Philipp Kraft

This module calculates scalars from timeseries to describe signature behaviour.

The signature behaviour indices are collected from different sources:

.. [WESMCM2015] *Uncertainty in hydrological signatures*
            by I. K. Westerberg and H. K. McMillan, 2015
            `DOI: 10.5194/hess-19-3951-2015 <https://doi.org/10.5194/hess-19-3951-2015>`_
.. [CLBGS2000] *Flow variables for ecological studies in temperate streams: groupings based on covariance*,
                B Clausen, BJF Biggs 2000
                `DOI:10.1016/S0022-1694(00)00306-1 <https://doi.org/10.1016/S0022-1694(00)00306-1>`_
.. [YADV2007] *Regionalization of constraints on expected watershed response behavior for improved
               predictions in ungauged basins*, M Yadev, T Wagener, H Gupta, 2007
               `DOI:10.1016/j.advwatres.2007.01.005<https://doi.org/10.1016/j.advwatres.2007.01.005>`_

All methods without further parameters in this module should follow
the following template, where "signature" is replaced with the actual name

>>> def get_signature(data, measurements_per_day=None):
...     return sum(data)

**Where:**
- `data` is the data time series as any sequence of floats
- `measurements_per_day` is the raster length of the timeseries in seconds.
  For daily time series, `measurements_per_day=86400`. Many signatures ignore the `measurements_per_day`parameter,
  but it needs to be provided to ensure a consistent interface

TODO: Check if `measurements_per_day` can be removed from the interface

If a method needs additional parameters, create a class for that method,
with the parameters as attributes of an instance and a __call__ method
with the same interface as the function above.

>>> class Signature(__BaseSignature):
...     def __init__(self, parameter):
...         self.parameter = parameter
...     def __call__(self, data, measurements_per_day=None):
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
    filled, using a linear approximation between the neighbors.
    Gaps at the beginning or end are filled with the first resp. last
    valid entry

    :param data: The timeseries data as a numeric sequence
    :return:
    """
    # All data indices
    x = np.arange(len(data))
    # Valid data indices
    xp = np.flatnonzero(np.isfinite(data))
    # Valid data
    fp = remove_nan(data)
    # Interpolate missing values
    return np.interp(x, xp, fp)


def summarize(data, step, f):
    """
    Summarizes data for step using function f

    Example: Get yearly minimum from a daily timeseries:

    >>> yearly_min = summarize(data, 365, np.min)

    :param data: the timeseries
    :param step: int the number of time steps to summarize
    :param f: a function to summarize, eg. np.min, np.max etc.
            Must except an arraylike as the only parameter and return a
            float
    :return: The summarized timeseries
    """
    if len(data) < step:
        return np.array([f(data)])
    return np.fromiter((f(data[i:i+step])
                        for i in range(0, len(data), step)),
                       count=len(data) // step, dtype=float)


class Quantile(object):
    """
    Calculates the <quantile>% percentile from a runoff time series.

    Used as a signature behaviour index by [WESMCM2015]_
    """
    def __init__(self, quantile):
        self.percentile = 100 - quantile
        self.__doc__ = _getdoc(type(self)).replace('<quantile>', '{:0.4g}'.format(self.percentile))

    def __call__(self, data, measurements_per_day=None):
        """
        Calculates the flow <quantile>% of the time exceeded from a runoff time series.

        :param data: The timeseries data as a numeric sequence
        :param measurements_per_day: Unused
        :return: quantile signature behaviour index
        """
        return np.percentile(remove_nan(data), self.percentile)

    def __repr__(self):
        return 'q({:0.2f}%)'.format(self.percentile)


get_q0_01 = Quantile(0.01)
get_q0_1 = Quantile(0.1)
get_q1 = Quantile(1)
get_q5 = Quantile(5)
get_q50 = Quantile(50)
get_q85 = Quantile(85)
get_q95 = Quantile(95)
get_q99 = Quantile(99)


def get_mean(data, measurements_per_day=None):
    """
    Calculates the mean from a runoff time series.

    Used as a signature behaviour index by [WESMCM2015]_ and [CLBGS2000]_

    :param data: The runoff timeseries data as a numeric sequence
    :param measurements_per_day: the measurements_per_day of the timeseries (unused)
    :return: A single number containing the signature behaviour index
    """
    return np.mean(remove_nan(data))


def get_skewness(data, measurements_per_day=None):
    """
    Skewness, i.e. the mean flow data divided by Q50.

    See: [CLBGS2000]_ and [WESMCM2015]_

    :param data: The runoff timeseries data as a numeric sequence
    :param measurements_per_day: the measurements_per_day of the timeseries (unused)
    :return: A single number containing the signature behaviour index

    """
    return get_mean(data) / get_q50(data)


def get_cv(data, measurements_per_day=None):
    """
    Coefficient of variation, i.e. standard deviation divided by mean flow

    See: [CLBGS2000]_

    :param data: The runoff timeseries data as a numeric sequence
    :param measurements_per_day: the measurements_per_day of the timeseries (unused)
    :return: A single number containing the signature behaviour index

    """
    return remove_nan(data).std() / get_mean(data)


def get_sfdc(data, measurements_per_day=None):
    """
    The slope in the middle part of the ﬂow duration curve, calculated between the 33rd and 66th
    streamﬂow percentiles

    [YADV2007]_:

    .. math::
        S_{fdc} = \\frac{\\ln Q_{66} - \\ln Q_{33}}{0.66 - 0.33}

    :math:`Q_{X}` is the X'th percentile of the normalized discharge :math:`Q / \\overline{Q_{mean}}`

    Used by:[WESMCM2015]_, [YADV2007]_,

    :param data: The runoff timeseries data as a numeric sequence
    :param measurements_per_day: the measurements_per_day of the timeseries (unused)
    :return: A single number containing the signature behaviour index

    """
    mean = get_mean(data)

    Q33 = Quantile(33)(data)/mean
    Q66 = Quantile(66)(data)/mean

    return (np.log(Q33) - np.log(Q66)) / (2/3 - 1/3)


def calc_baseflow(data, measurements_per_day=1):
    """
    Calculates the 5 day baseflow after Gustard et al 1992, p. 21
    See:
        Report No. 108, Low flow estimation in the United Kingdom, . Gustard, A. Bullock December 1992 and J. M. Dixon"
        http://nora.nerc.ac.uk/id/eprint/6050/1/IH_108.pdf

    :param data:
    :param measurements_per_day:
    :return:
    """
    period_length = 5 # days
    if measurements_per_day < 1:
        raise ValueError('At least a daily measurement frequency is needed to calculate baseflow')

    # Remove NaN values
    data = fill_nan(data)

    def irange(seq):
        """ Returns the indices of a sequence"""
        return range(len(seq))

    # Calculate daily mean
    daily_flow = summarize(data, measurements_per_day, np.mean)

    # Get minimum flow for each 5 day period (Step 1 in Gustard et al 1992)
    Q = summarize(daily_flow, period_length, np.min)


    def is_baseflow(i, Q):
        """
        Returns True if a 5 day period can be considered as baseflow
        :param i: Actual 5day period index
        :param Q: 5day period minimum values
        :return: True if Q[i] is a baseflow
        """
        if 0 < i < len(Q)-1:
            return Q[i] * 0.9 < min(Q[i - 1], Q[i + 1])
        else:
            return False

    # Get each 5 day period index, where the baseflow condition is fullfilled
    # (Step 2 in Gustard et al 1992)
    QB_pos = [i for i in irange(Q)
              if is_baseflow(i, Q)]

    QB_raw = Q[QB_pos]

    # get interpolated values for each minflow timestep (Step 3)
    QB_int = np.interp(irange(Q), QB_pos, QB_raw)

    # If QBi > Qi then QBi = Qi (Step 4)
    QB = np.where(QB_int > Q, Q, QB_int)

    # Return the baseflow interpolated to the data line
    # using a time axis t in the unit of period indices (eg 1/(5 days))
    t = np.linspace(0, len(QB) - 1, len(data))
    return np.interp(t, irange(QB), QB)


def get_bfi(data, measurements_per_day=1):
    """
    Returns the baseflow index after Gustard et al 1992

    See:
        Report No. 108, Low flow estimation in the United Kingdom, . Gustard, A. Bullock December 1992 and J. M. Dixon
        http://nora.nerc.ac.uk/id/eprint/6050/1/IH_108.pdf

    :param data: The runoff timeseries data as a numeric sequence
    :param measurements_per_day: the measurements_per_day of the timeseries in seconds, default is daily
    :return: A single number containing the signature behaviour index
    """

    # Calculates the timeseries for the baseflow follwing Gustard et al 1992, p. 20ff, Step 1-4
    baseflow = calc_baseflow(data, measurements_per_day)

    return baseflow.mean() / np.mean(data)


def flow_event(data, event_condition, *ec_args):
    """
    Returns the frequency and mean duration of events.

    Events can be eg. high flow, low flow or whatever can be determined from a single value
    of the timeseries. This function is used be get_qhf, get_qhd, get_qlf, get_qhd.

    In difference to [WESMCM2016]_ the frequency is in occurences per timestep and hence quite a small number (multiply with 365 to gain :math:`yr^{-1}`) and
    the mean duration is in days, if measurements_per_day is given. Without a step size the mean duration is in multiples of
    the timeseries measurements_per_day, whatever it is (5min, 1h, 1day, 1y...).

    :param data: the timeseries
    :param event_condition: a callable checking for a single value if an event is happening
    :param ec_args: Additional arguments for the event condition
    :return: frequency, mean duration
    """

    # A list of events. An event is characterized by its duration
    events = []
    actual_event = False
    for v in data:
        if event_condition(v, *ec_args):  # We have an event!
            if not actual_event:  # A new event starts
                events.append(1)
                actual_event = True
            else:  # A current event continues
                events[-1] += 1
        else:  # No event
            actual_event = False

    freq = len(events) / len(data)
    mean_duration = np.mean(events)

    return freq, mean_duration


def get_qhf(data, measurements_per_day=1):
    """
    Calculates the frequency of high flow events defined as :math:`Q > 9 \\cdot Q_{50}`

    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math:`day^{-1}` and not
    in :math:`yr^{-1}` and for the whole timeseries

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: frequency of event starts per day
    """

    def highflow(value, median):
        return value > 9 * median

    fq, md = flow_event(data, highflow, np.median(data))

    return fq * measurements_per_day


def get_qhd(data, measurements_per_day=1):
    """
    Calculates the mean duration of high flow events as :math:`Q > 9 \\cdot Q_{50}`
    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math:`day^{-1}` and not
    in :math:`yr^{-1}` and for the whole timeseries

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: mean duration of high flow events in days
    """

    def highflow(value, median):
        return value > 9 * median
    fq, md = flow_event(data, highflow, np.median(data))
    return md / measurements_per_day


def get_qlf(data, measurements_per_day=1):
    """
    Calculates the frequency of low flow events defined as
    :math:`Q < 0.2 \\cdot \\overline{Q_{mean}}`

    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math:`day^{-1}` and not
    in :math:`yr^{-1}` and for the whole timeseries

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: frequency of event starts per day
    """

    def lowflow(value, mean):
        return value < 0.2 * mean
    fq, md = flow_event(data, lowflow, np.mean(data))
    return fq * measurements_per_day


def get_qld(data, measurements_per_day=1):
    """
    Calculates the mean duration of of low flow events defined as
    :math:`Q < 0.2 \\cdot \\overline{Q_{mean}}`

    cf. [CLBGS2000]_, [WESMCM2015]_. The frequency is given as :math:`day^{-1}` and not
    in :math:`yr^{-1}` and for the whole timeseries

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: mean duration of high flow events in days
    """

    def lowflow(value, mean):
        return value < 0.2 * mean
    fq, md = flow_event(data, lowflow, np.mean(data))
    return md / measurements_per_day


def get_ac(data, measurements_per_day=1):
    """
    Calculates the autocorrelation for 1 day

    cf. [WESMCM2015]_

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: The autocorrelation with 1 day shift
    """

    front = fill_nan(data[measurements_per_day:])
    back = fill_nan(data[:-measurements_per_day])

    return np.corrcoef(front, back)[0, 1]


def get_qlv(data, measurements_per_day=1):
    """
    Calculates the low flow variability as low flow per median flow

    Here low flow (:math:`LF_{mean/yr}`) is defined as the mean of the minimum flow per year

    cf. [WESMCM2015]_

    .. math::
        Q_{LV} = \\frac{LF_{mean/yr}}{Q_{50}}

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: The lowflow variability
    """

    year = measurements_per_day * 365
    # Calculate mean annual low flow
    data = fill_nan(data)
    lf = np.mean(summarize(data, year, np.min))

    return lf / get_q50(data)


def get_qhv(data, measurements_per_day=1):
    """
    Calculates the low flow variability as low flow per median flow

    Here low flow (:math:`LF_{mean/yr}`) is defined as the mean of the minimum flow per year

    cf. [WESMCM2015]_

    .. math::
        Q_{LV} = \\frac{LF_{mean/yr}}{Q_{50}}

    :param data: the timeseries
    :param measurements_per_day: the measurements_per_day of the timeseries
    :return: The lowflow variability
    """

    year = measurements_per_day * 365
    # Calculate mean annual low flow
    data = fill_nan(data)
    lf = np.mean(summarize(data, year, np.max))

    return lf / get_q50(data)


