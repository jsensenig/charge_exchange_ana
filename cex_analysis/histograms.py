import cex_analysis.plot_utils as utils
from cex_analysis.histogram_data import HistogramData
import ROOT
from ROOT import TEfficiency, TLegend, TH1D
import awkward as ak
import plotting_utils
import numpy as np
from copy import deepcopy


class Histogram():
    def __init__(self, config):
        self.config = config

        # Create a map to hold all plot objects
        # Not necessary to initialize map but do it so keys are known and consistent
        # Each entry in the lists should be a tuple = (<object>, <legend>)
        # Each entry in the lists should be a dict = {<name>: [(<object>, <legend>)]}
        self.hist_map = {"efficiency": [], "hist": [], "stack": []}
        self.hist_data = []

    def get_hist_map(self):
        return self.hist_data

    # def merge_all_plots(self, hist_map, file):
    #     """
    #     Merge equivalent histograms from different threads
    #     :param file:
    #     :param hist_map: Map of histograms from different threads
    #     :return: Single merged histogram
    #     """
    #     for key in hist_map:
    #         # We can merge all hists the same way except for TEfficiency objects
    #         if key == "efficiency":
    #             self.merge_efficiency_list(hist_map[key])
    #         else:
    #             self.merge_hist_list(hist_map[key])

    @staticmethod
    def merge_hist_list(hist_list):
        # If no histograms, skip
        if len(hist_list) < 1:
            print("Empty histogram list.. returning")
            return None
        print("MERGING! ", hist_list[0].hist_type)
        # Get the first histogram and clear it, so we can merge into it
        # Note the `histogram` member is a tuple = (Histogram, Legend)
        if hist_list[0].hist_type == "stack":
            hmerge = hist_list[0].histogram[0]
            print(hmerge)
        else:
            hmerge = hist_list[0].histogram[0].Clone("hmerge")
            hmerge.Reset()

        # Unfortunately `merge()` requires TList input so create and fill one
        tlist = ROOT.TList()
        for hist in hist_list[1:]:
            tlist.Add(hist.histogram[0])

        # Merge all histograms and return the result
        if hmerge.Merge(tlist, ROOT.nullptr) == -1:
            print("Error merging histograms!")

        return hmerge

    @staticmethod
    def sum_hist_list(hist_list):
        # If no histograms, skip
        if len(hist_list) < 1:
            print("Empty histogram list.. returning")
            return None
        print("MERGING! ", hist_list[0].hist_type, " ", hist_list[0].histogram[0])
        # Get the first histogram and clear it, so we can merge into it
        # Note the `histogram` member is a tuple = (Histogram, Legend)
        # if hist_list[0].hist_type == "stack":
        #     hmerge = hist_list[0].histogram[0]
        #     print(hmerge)
        # else:
        #     hmerge = hist_list[0].histogram[0].Clone("hmerge")
        #     hmerge.Reset()

        # Unfortunately `merge()` requires TList input so create and fill one
        # tlist = ROOT.TList()
        # for hist in hist_list[1:]:
        #     tlist.Add(hist.histogram[0])
        hsum = hist_list[0].histogram[0]
        print(hsum)
        for hist in hist_list[1:]:
            hsum.Add(hist.histogram[0])
        print(hsum)
        # Merge all histograms and return the result
        # if hmerge.Merge(tlist, ROOT.nullptr) == -1:
        #     print("Error merging histograms!")

        return hsum

    def merge_efficiency_list(self, eff_list):
        """
        Leverages the builtin "+=" to merge all TEfficiency objects from the same process
        :param eff_list: Map of all TEfficiency objects
        :return: Merged TEfficiency object
        """
        if len(eff_list) < 1:
            print("Empty histogram list.. returning")
            return None

        merged_eff = TEfficiency()
        for eff in eff_list:
            merged_eff += eff.histogram[0]

        return merged_eff

    def init_plotting(self, plot_config):
        """
        Initialize the plot objects, histograms
        :param plot_config:
        :return: Initialize success/failure
        """
        init_success = True

        return init_success

    def test_th1(self):
        h = TH1D("t", "t;x;c", 5, 0, 5)

    def plot_particles(self, x, cut):
        legend = utils.legend_init_right()
        # Get the config to create the plot for this cut
        name, title, bins, upper_lim, lower_lim = self.config["cut_plots"][cut]
        hist = TH1D(name, title, bins, upper_lim, lower_lim)

        # Just a loop in c++ which does hist.Fill()
        # nullptr sets weights = 1
        if isinstance(x, ak.Array):
            hist.FillN(len(x), ak.to_numpy(x), ROOT.nullptr)
        elif isinstance(x, np.ndarray):
            hist.FillN(len(x), x, ROOT.nullptr)
        else:
            print("Unknown array type!")

        legend.AddEntry(name)

        # Store this hist in our master map as HistogramData class
        self.hist_data.append(HistogramData("hist", name, hist))

    def plot_particles_stack(self, x, x_pdg, cut, precut):
        """
        Make stacked plot of a given variable with each stack corresponding to a PDG
        :param precut: Is this pre or Post cut plot
        :param x: Array of single variable either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param x_pdg: Array of PDG codes with either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param cut: str Cut name
        :return:
        """
        if not isinstance(x, ak.Array) or not isinstance(x_pdg, ak.Array):
            print("Must use Awkward Arrays to plot!")

        c = ROOT.TCanvas()
        stack = ROOT.THStack()
        legend = utils.legend_init_right()

        # The name and binning should be the same for all particles
        name, title, bins, upper_lim, lower_lim = self.config["cut_plots"][cut]
        if precut:
            name = "precut_" + name
            title = "PreCut-" + title
        else:
            name = "postcut_" + name
            title = "PostCut-" + title

        for pdg in self.config["stack_pdg_list"]:

            hstack = TH1D(name, title, bins, lower_lim, upper_lim)

            # Before plotting we flatten from 2D array shape=(<num event>, <num daughters>) to
            # 1D array shape=(<num event>*<num daughters>)
            pdg_filtered_array = plotting_utils.daughter_by_pdg(ak.flatten(x, axis=None),
                                                                ak.flatten(x_pdg, axis=None), pdg)

            if len(pdg_filtered_array) < 1:
                continue

            # Now use the standard fill function
            hstack.FillN(len(pdg_filtered_array), pdg_filtered_array, ROOT.nullptr)

            utils.set_hist_colors(hstack, utils.colors.get(utils.pdg2string.get(pdg, "Other"), 1),
                                  utils.colors.get(utils.pdg2string.get(pdg, "Other"), 1))
            stack.Add(hstack, "HIST")
            legend.AddEntry("hstack", utils.pdg2string.get(pdg, "other"))

        stack.Draw()

        # Store this hist in our master map as a HistogramData class
        self.hist_data.append(HistogramData("stack", name, deepcopy(stack)))

        return

###################################################

    def plot_particles(self, event):
        pass


    def set_hist_lim(hist):
        hist.SetAxisRange(0, 2, "X")

    def print_count(count):
        for p in count:
            print(p, ":", count[p])

    def print_cut_report(pdg, total, passed):
        print(f"{pdg}: (passed/total = {passed}/{total})  (Eff = {100 * (passed / total):.1f}%)")

    def hist_help(file, hist, color, fill=True):
        h = file.Get(hist)
        h.SetLineColor(color)
        h.SetFillColor(color)
        if not fill:
            h.SetFillColor(False)
        return h




