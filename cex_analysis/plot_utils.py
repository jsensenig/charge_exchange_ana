import ROOT

color_text = '\033[31mpositron : light_red ' \
             '\033[0m, \033[34m electron : blue ' \
             '\033[0m, \033[32m proton : green ' \
             '\033[0m, \033[35m pion : magenta ' \
             '\033[0m, muon : black'

colors = {'positron': 46,
          'electron': 49,
          'proton': 8,
          'pion': 64,
          'pion_minus': 67,
          'anti_muon': 37,
          'muon': 38,
          'kaon': 53,
          'gamma': 94,
          'other': 29}

pdg2string = {-11: 'positron',
              11: 'electron',
              2112: 'neutron',
              2212: 'proton',
              321: 'kaon',
              -211: 'pion_minus',
              211: 'pion',
              -13: 'anti_muon',
              13: 'muon',
              22: 'gamma',
              0: "other"}


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