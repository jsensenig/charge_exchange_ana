import copy
import numpy as np


class Hist1d:
    def __init__(self, num_bins, bin_range):
        self.num_bins = num_bins
        self.bin_range = bin_range
        _, self.bins = np.histogram([], bins=self.num_bins, range=self.bin_range)
        self.hist = None
        self.legend = []

    def __add__(self, other):
        if not isinstance(other, Hist1d):
            raise TypeError
        obj_copy = copy.deepcopy(self)
        obj_copy.hist = self.hist + other.hist

        return obj_copy

    def fill_hist(self, x, legend=None, weights=None):
        self.hist, _ = np.histogram(x, bins=self.num_bins, range=self.bin_range, weights=weights)

        if legend is not None:
            self.legend.append(legend)

    def set_legend(self, legend):
        self.legend = legend

    def get_hist(self):
        return self.hist, self.bins


class HistStack(Hist1d):
    def __init__(self, num_bins, bin_range):
        super().__init__(num_bins=num_bins, bin_range=bin_range)
        self.stack = {'bins': self.bins, 'hists': []}
        self.legend = []

    def __add__(self, other):
        if not isinstance(other, HistStack):
            raise TypeError
        obj_copy = copy.deepcopy(self)
        obj_copy.stack['hists'] = [h1 + h2 for h1, h2 in zip(self.stack['hists'], other.stack['hists'])]

        return obj_copy

    def fill_stack(self, x, legend=None, weights=None):
        self.fill_hist(x=x, legend=legend, weights=weights)
        self.stack['hists'].append(self.hist)

    def set_legend(self, legend):
        self.set_legend(legend=legend)

    def get_hist(self):
        return self.stack['hists'], self.stack['bins']


class HistEff(Hist1d):
    def __init__(self, num_bins, bin_range):
        super().__init__(num_bins=num_bins, bin_range=bin_range)
        self.efficiency = {'bins': self.bins, 'total': None, 'passed': None}
        self.legend = []

    def __add__(self, other):
        if not isinstance(other, HistEff):
            raise TypeError
        obj_copy = copy.deepcopy(self)
        obj_copy.efficiency['total'] = self.efficiency['total'] + other.efficiency['total']
        obj_copy.efficiency['passed'] = self.efficiency['passed'] + other.efficiency['passed']
        return obj_copy

    def fill_total(self, x, legend=None, weights=None):
        self.fill_hist(x=x, legend=legend, weights=weights)
        self.efficiency['total'] = self.hist

    def fill_passed(self, x, legend=None, weights=None):
        self.fill_hist(x=x, legend=legend, weights=weights)
        self.efficiency['passed'] = self.hist

    def set_legend(self, legend):
        self.set_legend(legend=legend)

    def get_hist(self):
        return self.efficiency['passed'], self.efficiency['total'], self.efficiency['bins']

    def get_efficiency_hist(self):
        if self.efficiency['passed'] is not None and self.efficiency['passed'] is not None:
            eff = self.efficiency['passed'] / self.efficiency['total']
            eff[self.efficiency['total'] == 0] = 0
            return eff, self.efficiency['bins']
        else:
            print("No Passed or Total histograms!")
            raise ValueError


class HistogramData:
    def __init__(self, hist_type, hist_name, histogram, precut):
        self.hist_type = hist_type    # Histogram type e.g. stack, hist, efficiency
        self.hist_name = self.generate_name(hist_name=hist_name, precut=precut)
        self.histogram = histogram    # <Histogram Object>

    def generate_name(self, hist_name, precut):
        type_name = 'eff_' if self.hist_type == 'efficiency' else self.hist_type + '_'
        if precut:
            name = 'precut_' + type_name + hist_name
        else:
            name = 'postcut_' + type_name + hist_name

        return name


"""
Helper functions to sort and access the HistogramData dataclass
"""


# Sorting function by type
def sort_hist_type(e):
    return e.hist_type


# Sorting function by name
def sort_hist_name(e):
    return e.hist_name


# Return a list of a specified hist_type
def get_hist_type_list(hlist, select_type):
    return [h for h in hlist if h.hist_type == select_type]


# Return a list of matching hist_names
def get_select_hist_name_list(hlist, select_name):
    return [h for h in hlist if h.hist_name == select_name]


# Return a list of matching unique hist_names
def get_hist_name_list(hlist):
    return set([h.hist_name for h in hlist])
