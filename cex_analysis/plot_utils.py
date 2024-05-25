import ROOT
import numpy as np

color_text = '\033[31mpositron : light_red ' \
             '\033[0m, \033[34m electron : blue ' \
             '\033[0m, \033[32m proton : green ' \
             '\033[0m, \033[35m pion : magenta ' \
             '\033[0m, muon : black'

# ROOT color palette: https://root-forum.cern.ch/t/what-is-your-best-way-to-increment-colors/13809/2
colors = {-11:  46,  # red
          11:   49,  # brown
          2212: 8,   # green
          211:  64,  # blue
          -211: 67,  # cyan
          -13:  40,  # green-brown
          13:   38,  # light-blue
          321:  9,   # purple
          22:   92,  # orange
          0:    29}  # grey-green

proc_colors = {'pion_elastic': 46,
               'pion_and_pi0': 49,
               'pi0_production': 8,
               'single_charge_exchange': 64,
               'double_charge_exchange': 67,
               'pion_production': 38,
               'absorption': 53,
               'quasi_elastic': 94,
               'charged_neutral_pion_production': 130,
               'mctruth_charged_neutral_pion': 91,
               'mcreco_charged_neutral_pion': 24,
               'other': 29}

pdg2string = {-11:  'e+',
              11:   'e-',
              2112: 'n',
              2212: 'p',
              321:  'K+',
              -211: '#pi-',
              211:  '#pi+',
              111:  '#pi0',
              -13:  '#mu+',
              13:   '#mu-',
              22:   '#gamma',
              0:    'other'}


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
