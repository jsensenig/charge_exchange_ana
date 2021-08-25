from dataclasses import dataclass
import typing


@dataclass
class HistogramData:

    hist_type: str           # Histogram type e.g. stack, hist, efficiency
    hist_name: str           # Histogram name
    histogram: typing.Any    # Tuple of (<Histogram Object>, <Legend Object>)


"""
Helper functions to sort and access the HistoramData dataclass
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


# Return a list of matching  unique hist_names
def get_hist_name_list(hlist):
    return set([h.hist_name for h in hlist])
