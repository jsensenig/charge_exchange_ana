import cex_analysis.plot_utils as utils
import ROOT
from ROOT import TEfficiency, TLegend, TH1D
import awkward as ak
import plotting_utils


class Histogram:
    def __init__(self, config):
        self.config = config

        # Create a map to hold all plot objects
        # Not necessary to initialize map but do it so keys are known and consistent
        # Each entry in the lists should be a tuple = (<object>, <legend>)
        self.hist_map = {"efficiency": [], "hist": [], "stack": []}

    def init_eff(self):
        cut_eff = {}
        for cut in self.config["cut_list"]:
            cut_eff[cut] = TEfficiency("eff", "Eff;x;#epsilon", 40, 0, 80)

        return cut_eff

    def init_hists(self):
        return self.hist_map

    def merge_hists(self, hist_list):
        """
        Merge equivalent histograms from different threads
        :param hist_list: List of histograms from different threads
        :return: Single merged histogram
        """
        # Get a histogram and clear it, so we can merge into it
        hmerge = hist_list[0].Clone("hmerge")
        hmerge.Reset()

        # Unfortunately `merge()` requires TList input so create and fill one
        h_list = ROOT.TList()
        for h in hist_list:
            h_list.Add(h)

        # Merge all histograms and return the result
        hmerge.Merge(h_list)
        return hmerge

    def merge_eff_hist(self, eff_list):
        """
        Leverages the builtin "+=" to merge all TEfficiency objects from the same process
        :param eff_list: Map of all TEfficiency objects
        :return: Merged TEfficiency object
        """
        merged_eff = TEfficiency()
        for eff in eff_list:
            merged_eff += eff

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
        # Get the config to create the plot for this cut
        name, title, bins, upper_lim, lower_lim = self.config["cut_plots"][cut]
        hist = TH1D(name, title, bins, upper_lim, lower_lim)

        # Just a loop in c++ which does hist.Fill()
        # nullptr sets weights = 1
        hist.FillN(len(x), ak.to_numpy(x), ROOT.nullptr)

        # Store this hist in our master map
        self.hist_map["hist"].append(hist)

    def plot_particles_stack(self, x, x_pdg, cut, precut):
        """
        Make stacked plot of a given variable with each stack corresponding to a PDG
        :param precut: Is this pre or Post cut plot
        :param x: Array of single variable either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param x_pdg: Array of PDG codes with either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param cut: str Cut name
        :return:
        """
        c = ROOT.TCanvas()
        stack = ROOT.THStack()
        legend = utils.legend_init_right()

        for pdg in self.config["stack_pdg_list"]:
            name, title, bins, upper_lim, lower_lim = self.config["cut_plots"][cut]
            if precut:
                name = "precut_" + name
                title = "PreCut-" + title
            else:
                name = "postcut_" + name
                title = "PostCut-" + title

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

        self.hist_map["stack"].append((stack, legend))

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




