import ROOT
import numpy as np

color_text = '\033[31mpositron : light_red ' \
             '\033[0m, \033[34m electron : blue ' \
             '\033[0m, \033[32m proton : green ' \
             '\033[0m, \033[35m pion : magenta ' \
             '\033[0m, muon : black'

code2string = {-11:  '$e^+$',
                11:   'e^-$',
                2112: 'n',
                2212: 'p',
                321:  '$K^+$',
                -211: '$\\pi^-$',
                211:  '$\\pi^+$',
                111:  '$\\pi^0$',
               -13:  '$\\mu^+$',
               13:   '$\\mu^-$',
               22:   '$\\gamma$',
               0:    'other',
               10001: 'pion_elastic',
               10002: 'pion_and_pi0',
               10003: 'pi0_production',
               10004: 'single_charge_exchange',
               10005: 'double_charge_exchange',
               10006: 'pion_production',
               10007: 'absorption',
               10008: 'quasi_elastic',
               10009: 'charged_neutral_pion_production',
               10010: 'mctruth_charged_neutral_pion',
               10011: 'mcreco_charged_neutral_pion',
               10012: 'other'}

string2code = {code2string[-11]: -11,
               code2string[11]: 11,
               code2string[2112]: 2112,
               code2string[2212]: 2212,
               code2string[321]: 321,
               code2string[-211]: -211,
               code2string[211]: 211,
               code2string[111]: 111,
               code2string[-13]: -13,
               code2string[13]: 13,
               code2string[22]: 22,
               code2string[0]: 0,
               code2string[10001]: 10001,
               code2string[10002]: 10002,
               code2string[10003]: 10003,
               code2string[10004]: 10004,
               code2string[10005]: 10005,
               code2string[10006]: 10006,
               code2string[10007]: 10007,
               code2string[10008]: 10008,
               code2string[10009]: 10009,
               code2string[10010]: 10010,
               code2string[10011]: 10011,
               code2string[10012]: 10012}

# matplotlib colors
colors = {-11:  'indianred',  # red
          11:   'orchid',  # brown
          2212: 'seagreen',   # green
          211:  'royalblue',  # blue
          -211: 'paleturquoise',  # cyan
          -13:  'peru',  # green-brown
          13:   'cadetblue',  # light-blue
          321:  'mediumorchid',   # purple
          22:   'goldenrod',  # orange
          0:    'olivedrab',  # grey-green
        10001: 'indianred',
        10002: 'orchid',
        10003: 'seagreen',
        10004: 'royalblue',
        10005: 'paleturquoise',
        10006: 'peru',
        10007: 'cadetblue',
        10008: 'darkmagenta',
        10009: 'goldenrod',
        10010: 'coral',
        10011: 'lightpink',
        10012: 'olivedrab'}

# ROOT color palette: https://root-forum.cern.ch/t/what-is-your-best-way-to-increment-colors/13809/2
# colors = {-11:  46,  # red
#           11:   49,  # brown
#           2212: 8,   # green
#           211:  64,  # blue
#           -211: 67,  # cyan
#           -13:  40,  # green-brown
#           13:   38,  # light-blue
#           321:  9,   # purple
#           22:   92,  # orange
#           0:    29}  # grey-green

# proc_colors = {'pion_elastic': 46,
#                'pion_and_pi0': 49,
#                'pi0_production': 8,
#                'single_charge_exchange': 64,
#                'double_charge_exchange': 67,
#                'pion_production': 38,
#                'absorption': 53,
#                'quasi_elastic': 94,
#                'charged_neutral_pion_production': 130,
#                'mctruth_charged_neutral_pion': 91,
#                'mcreco_charged_neutral_pion': 24,
#                'other': 29}

# proc_colors = {'pion_elastic': 'indianred',
#                'pion_and_pi0': 'orchid',
#                'pi0_production': 'seagreen',
#                'single_charge_exchange': 'royalblue',
#                'double_charge_exchange': 'paleturquoise',
#                'pion_production': 'peru',
#                'absorption': 'cadetblue',
#                'quasi_elastic': 'darkmagenta',
#                'charged_neutral_pion_production': 'goldenrod',
#                'mctruth_charged_neutral_pion': 'coral',
#                'mcreco_charged_neutral_pion': 'lightpink',
#                'other': 'olivedrab'}

# Helper to set the histogram colors
def set_hist_colors(hist, lcolor, fcolor):
    hist.SetLineColor(lcolor)
    hist.SetFillColor(fcolor)


# Function to set legend template for right-side of canvas
def legend_init_right():
    legend = ROOT.TLegend(.65, .55, .85, .85)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.030)

    return legend


# Function to set legend template for left-side of canvas
def legend_init_left():
    legend = ROOT.TLegend(.15, .55, .35, .85)
    legend.SetBorderSize(0)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    legend.SetTextSize(0.030)

    return legend


def histogram_constructor(title, bin_arrays, ndim):

    if ndim == 1:
        return ROOT.TH1D(title + "_hist", title, len(bin_arrays[0])-1, bin_arrays[0])
    elif ndim == 2:
        return ROOT.TH2D(title + "_hist", title, bin_arrays[0], bin_arrays[1])
    elif ndim == 3:
        return ROOT.TH3D(title + "_hist", title, bin_arrays[0], bin_arrays[1], bin_arrays[2])
    else:
        print("Unsupported number of dimensions! Only ndim 1-3 supported.")
        print("Requested ndim =", ndim)
        raise ValueError


def bin_centers_np(bins):
    return (bins[:-1] + bins[1:]) / 2.


def bin_width_np(bins):
    return (bins[1] - bins[0])
