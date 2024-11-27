import ROOT
import numpy as np

color_text = '\033[31mpositron : light_red ' \
             '\033[0m, \033[34m electron : blue ' \
             '\033[0m, \033[32m proton : green ' \
             '\033[0m, \033[35m pion : magenta ' \
             '\033[0m, muon : black'

code2prettystring = {-11:  '$e^+$',
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
               10001: '$\\pi$ Elastic',
               10002: '$\\pi^{\\pm}$ + $\\pi^{0}$',
               10003: '$\\pi^{0}$ Production',
               10004: 'Single CeX',
               10005: 'Double CeX',
               10006: '$\\pi^{\\pm}$ Production',
               10007: 'Absorption',
               10008: 'QE',
               10009: 'charged_neutral_pion_production',
               10010: 'mctruth_charged_neutral_pion',
               10011: 'mcreco_charged_neutral_pion',
               10012: 'Other',
               10013: '$\\pi^{\\pm, 0}$ Production',
               10014: 'Daughter 1$\\pi^{0}$ Bkgd',
               10015: 'Daughter >1$\\pi^{0}$ Bkgd',
               10016: "Daughter Other Bkgd",
               10017: 'Daughter 0$\\pi^{0}$ Bkgd',
               10018: 'MisId Beam',
               10019: 'Other',
               10020: 'MisId $\\pi^{\\pm}$',
               10021: 'MisId $p$',
               10022: 'MisId $\\mu^{+}$',
               10023: 'MisId $e$/$\\gamma$',
               10024: '$\\pi^{\\pm}$',
               10025: '$\\mu^{+}$',
               10026: 'Other',
               10027: 'Beam Background',
               10028: 'Proton Inelastic',
               10029: 'Proton CeX',
               10030: 'Proton 0$\\pi^{0}$ Bkgd',
               10031: 'Proton 1$\\pi^{0}$ Bkgd',
               10032: 'Proton n$\\pi^{0}$ Bkgd',
               10033: 'Proton 0$\\pi$ Bkgd',
               10034: 'Proton Beam Bkgd',
               10035: 'Proton Other Bkgd'}

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
               10012: 'other',
               10013: 'all_pion_production',
               10014: 'daughter_one_pi0_bkgd',
               10015: "daughter_n_pi0_bkgd",
               10016: "daughter_other_bkgd",
               10017: 'daughter_zero_pi0_bkgd',
               10018: 'misid_beam',
               10019: 'simple_other',
               10020: 'misid_pion',
               10021: 'misid_proton',
               10022: 'misid_muon',
               10023: 'misid_electron_gamma',
               10024: 'matched_pion',
               10025: 'matched_muon',
               10026: 'beam_other',
               10027: 'beam_bkgd',
               10028: 'proton_inelastic',
               10029: 'proton_charge_exchange',
               10030: 'proton_zero_pi0_bkgd',
               10031: 'proton_one_pi0_bkgd',
               10032: 'proton_n_pi0_bkgd',
               10033: 'proton_zero_pion_bkgd',
               10034: 'proton_beam_bkgd',
               10035: 'proton_other_bkgd'}


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
               code2string[10012]: 10012,
               code2string[10013]: 10013,
               code2string[10014]: 10014,
               code2string[10015]: 10015,
               code2string[10016]: 10016,
               code2string[10017]: 10017,
               code2string[10018]: 10018,
               code2string[10019]: 10019,
               code2string[10020]: 10020,
               code2string[10021]: 10021,
               code2string[10022]: 10022,
               code2string[10023]: 10023,
               code2string[10024]: 10024,
               code2string[10025]: 10025,
               code2string[10026]: 10026,
               code2string[10027]: 10027,
               code2string[10028]: 10028,
               code2string[10029]: 10029,
               code2string[10030]: 10030,
               code2string[10031]: 10031,
               code2string[10032]: 10032,
               code2string[10033]: 10033,
               code2string[10034]: 10034,
               code2string[10035]: 10035}

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
        10012: 'olivedrab',
        10013: 'peru',
        10014: 'goldenrod',
        10015: 'indianred',
        10016: 'olivedrab',
        10017: 'seagreen',
        10018: 'darkmagenta',
        10019: 'olivedrab',
        10020: 'indianred',
        10021: 'paleturquoise',
        10022: 'goldenrod',
        10023: 'olivedrab',
        10024: 'royalblue',
        10025: 'peru',
        10026: 'olivedrab',
        10027: 'peru',
        10028: 'paleturquoise',
        10029: 'goldenrod',
        10030: 'olivedrab',
        10031: 'royalblue',
        10032: 'peru',
        10033: 'olivedrab',
        10034: 'peru',
        10035: 'indianred'}



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
    return bins[1] - bins[0]
