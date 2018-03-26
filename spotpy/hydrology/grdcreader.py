"""
A class to represent runoff data from the GRDC

:author: Philipp Kraft

Data provided by the GRDC comes in a simple ASCII file format
The class in this module simplifies data loading and handling


"""
from datetime import datetime as dt
from datetime import timedelta as td
import numpy as np


def _capitalize(words):
    return ' '.join(s.capitalize() for s in words.split())


class GRDCstation:
    @staticmethod
    def __read_item(line, keyword, transform=None):
        """
        Reads an item from a line and returns it transformed
        by the callable transform. If the item is not in that line returns None
        :param line: String with keyword and content, separated with a colon
        :param keyword: Keyword to look for `line.startswith('# ' + item)`
        :param transform: callable to transform the content
        :return:
        """
        transform = transform or str
        if line.startswith('# ' + keyword):
            return transform(line.split(':', 1)[1].strip())

    def __init__(self):

        self.id = None
        self.river = None
        self.name = None
        self.country = None
        self.lon = None
        self.lat = None
        self.area = None
        self.altitude = None

        self.t = np.array([], dtype=dt)
        self.Q = np.array([], dtype=float)
        self.years = self.__YearSelector(self)

    @classmethod
    def load(cls, filename):
        """
        Loads a station from the GRDC file format

        :param filename: Filename for GRDC station data
        :return: New GRDC station
        """

        grdc = cls()

        for line in open(filename):
            if not line.startswith('#'):
                break
            grdc.id = cls.__read_item(line, 'GRDC-No.', int) or grdc.id
            grdc.river = cls.__read_item(line, 'River', _capitalize) or grdc.river
            grdc.name = cls.__read_item(line, 'Station', _capitalize) or grdc.name
            grdc.country = cls.__read_item(line, 'Country') or grdc.country
            grdc.lon = cls.__read_item(line, 'Longitude', float) or grdc.country
            grdc.lat = cls.__read_item(line, 'Latitude', float) or grdc.country
            grdc.area = cls.__read_item(line, 'Catchment area', float) or grdc.area
            grdc.altitude = cls.__read_item(line, 'Altitude', float) or grdc.altitude

        data = np.genfromtxt(filename, delimiter=';', skip_header=36,
                             usecols=(0, 2), comments='#',
                             dtype=[('date', 'datetime64[s]'), ('Q', float)])

        grdc.t = data['date'].astype(dt)
        grdc.Q = data['Q']
        # Transform no data to nan
        grdc.Q[grdc.Q == -999.0] = np.nan

        return grdc

    def __len__(self):
        return len(self.Q)

    def __repr__(self):
        return 'GRDC(id={}, name={}, river={})'.format(self.id, self.name, self.river)

    @property
    def Q_mm(self):
        # m³/s -> mm/day = 86400 s/day * Q m³/s / A km² * 1e-6 km²/m² * 1e3 mm/m
        return self.Q * 86400 / self.area * 1e-3

    @property
    def start(self):
        return self.t[0]

    @property
    def end(self):
        return self.t[-1] + td(days=1)

    @staticmethod
    def __import_plt():
        try:
            import pylab as plt
        except ImportError:
            raise ImportError('For plotting, install python package matplotlib')
        return plt

    def plot(self, style='-', **kwargs):
        """
        plots the runoff timeseries
        :param style: Style for the plot
        :param kwargs: kwargs for pylab.plot
        :return: matplotlib.Line object
        """
        plt = self.__import_plt()
        return plt.plot_date(plt.date2num(self.t), self.Q, style, **kwargs)[0]

    def plot_fdc(self, style='-', **kwargs):
        """
        Plots the flow duration curve
        :param style: line style
        :param kwargs: keyword arguments for matplotlib.plot function
        :return: matplotlib.Line object
        """
        plt = self.__import_plt()

        q_sort = np.sort(self.Q)
        x = np.linspace(1, 0, len(self))
        plt.semilogy(x, q_sort, style, **kwargs)

    def metadata_copy(self):
        """
        Returns a new GRDC station object without data but with the same metadata
        :return: A GRDCstation object
        """
        copy = type(self)()
        copy.id = self.id
        copy.name = self.name
        copy.river = self.river
        copy.altitude = self.altitude
        copy.area = self.area
        copy.country = self.country
        copy.lat = self.lat
        copy.lon = self.lon
        copy.t = []
        copy.Q = []

        return copy

    def slice(self, start, stop, step=None):
        """
        Returns a part of the GRDC dataset
        :param start: start index (int) or start date (datetime.datetime)
        :param stop: index after end or date after end
        :param step: Step size (ignored for datetime index)
        :return: GRDCStation with sliced timeseries
        """
        copy = self.metadata_copy()
        if type(start) is int and type(stop) is int:
            copy.t = self.t[start:stop:step]
            copy.Q = self.Q[start:stop:step]
        else:
            try:
                take = (self.t >= start) & (self.t < stop)
            except (ValueError, TypeError) as e:
                raise type(e)('Only int or datetime objects can be used to get periods from the data set')
            copy.t = self.t[take]
            copy.Q = self.Q[take]

        return copy

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        else:
            return self.Q[item]

    class __YearSelector:
        def __init__(self, station):
            self.station = station

        def __getitem__(self, item):
            if isinstance(item, slice):
                return self.station.slice(dt(item.start, 1, 1), dt(item.stop, 1, 1))
            else:
                return self.station.slice(dt(item, 1, 1), dt(item+1, 1, 1))

        def __repr__(self):
            return '{}:{}'.format(self.station.start.year, self.station.end.year)

