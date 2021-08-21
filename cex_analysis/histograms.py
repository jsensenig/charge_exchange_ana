import numpy as np
import cex_analysis.utils as utils
import ROOT as rt
from ROOT import TEfficiency, TLegend, TH1D
from abc import abstractclassmethod, abstractmethod


class Histogram:
    def __init__(self, config):
        self.config = config

    def init_eff(self):
        cut_eff = {}
        for cut in self.config["cut_list"]:
            cut_eff[cut] = TEfficiency("eff", "Eff;x;#epsilon", 40, 0, 80)

        return cut_eff

    def init_hists(self):
        hist_map = {}
        for hist in self.config["hist_list"]:
            hist_map[hist] = TH1D("eff", "Eff;x;#epsilon", 40, 0, 80)

        return hist_map

    def merge_hists(self, hist_list):
        """
        Merge histograms from different threads but the same process
        :param hist_list: List of histograms from different threads
        :return: Single merged histogram
        """
        # Get a histogram and clear it, so we can merge into it
        hmerge = hist_list[0].Clone("hmerge")
        hmerge.Reset()

        # Merge requires TList input so create and fill it
        h_list = rt.TList()
        for h in hist_list:
            h_list.Add(h)

        # Merge all histograms and return the result
        hmerge.Merge(h_list)
        return hmerge

    def merge_eff_hist(self, eff_list):
        """
        Leverages the builtin "+=" to merge all TEfficiency objects from the same process
        :param eff_map: Map of all TEfficiency objects
        :return: Merged TEfficiency object
        """

        merged_eff = TEfficiency()
        for eff in eff_list:
            merged_eff += eff

        return merged_eff


###################################################

    def plot_particles(self, event):
        pass

    # Helper to set the histogram colors
    def set_hist_colors(hist, lcolor, fcolor):
        hist.SetLineColor(lcolor)
        hist.SetFillColor(fcolor)

    def set_hist_lim(hist):
        hist.SetAxisRange(0, 2, "X")

    def print_count(count):
        for p in count:
            print(p, ":", count[p])

    def print_cut_report(pdg, total, passed):
        print(f"{pdg}: (passed/total = {passed}/{total})  (Eff = {100 * (passed / total):.1f}%)")

    # Function to set legend template for right-side of canvas
    def legend_init_right(self):
        legend = TLegend(.65, .55, .85, .85)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.030)

        return legend

    # Function to set legend template for left-side of canvas
    def legend_init_left():
        legend = TLegend(.15, .55, .35, .85)
        legend.SetBorderSize(0)
        legend.SetFillColor(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(0.030)

        return legend

    def hist_help(file, hist, color, fill=True):
        h = file.Get(hist)
        h.SetLineColor(color)
        h.SetFillColor(color)
        if not fill:
            h.SetFillColor(False)
        return h

    def create_hist_stack(tree, cuts, pdg_list, var_string, bins, llim, ulim, legend):

        stacks = rt.THStack()
        count = {}

        for p in pdg_list:
            hist_name = utils.pdg2string[p] + ';Count;' + var_string
            hb = TH1D('hb', hist_name, bins, llim, ulim)

            count[utils.pdg2string[p]] = tree.Draw(var_string + " >> hb", "reco_beam_true_byHits_PDG==" + str(p) + cuts)
            h1 = rt.gROOT.FindObject("hb")
            set_hist_colors(h1, utils.colors[utils.pdg2string[p]], utils.colors[utils.pdg2string[p]])
            stacks.Add(h1, "HIST")
            legend.AddEntry("hb", utils.pdg2string[p])

        stacks.Draw()

        return stacks, legend

    def create_hist_stack_df(df, cuts, pdg_list, var_string, bins, llim, ulim, legend):

        stacks = rt.THStack()

        for p in pdg_list:
            hist_name = utils.pdg2string[p] + ';Count;' + var_string

            h1 = df.Filter(cuts).Filter("reco_beam_true_byHits_PDG==" + str(p)).Histo1D(
                ("hb", hist_name, bins, llim, ulim), var_string)
            set_hist_colors(h1, utils.colors[utils.pdg2string[p]], utils.colors[utils.pdg2string[p]])
            stacks.Add(h1, "HIST")
            legend.AddEntry(h1, utils.pdg2string[p])

        stacks.Draw()

        return stacks, legend


