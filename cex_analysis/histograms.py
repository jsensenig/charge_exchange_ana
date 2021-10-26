import cex_analysis.plot_utils as utils
from cex_analysis.histogram_data import HistogramData
from cex_analysis.true_process import TrueProcess
import ROOT
import awkward as ak
import plotting_utils
import numpy as np


class Histogram:
    def __init__(self, config):
        self.config = config
        self.hist_config = None

        # We don't want any plots drawn while running so set ROOT to Batch mode
        ROOT.gROOT.SetBatch(True)
        print("Setting ROOT to Batch mode!")

        # Create a list to hold all plot objects
        self.hist_data = []

        # Get the true process list
        self.true_process_list = TrueProcess.get_process_list()

        # Compile the C++ we will use for speed
        self.compile_cpp_helpers()

    def get_hist_map(self):
        return self.hist_data

    def configure_hists(self, config):
        self.hist_config = config

    @staticmethod
    def compile_cpp_helpers():

        # Fill a histogram with strings
        ROOT.gInterpreter.ProcessLine("""
            void fill_hist_th1i_string( std::string proc, TH1I *hist, int num ) {
                for(int i = 0; i < num; i++) hist->Fill(proc.c_str(), 1);
           }
        """)

    @staticmethod
    def merge_hist_list(hist_list):
        # If no histograms, skip or if one just return it
        if len(hist_list) < 1:
            return None
        elif len(hist_list) == 1:
            return hist_list[0].histogram

        # Sum all histograms into first one
        hsum = hist_list[0].histogram
        for hist in hist_list[1:]:
            hsum.Add(hist.histogram)

        return hsum

    @staticmethod
    def merge_stack_list(stack_list):
        # If no histograms, skip or if one just return it
        if len(stack_list) < 1:
            return None
        elif len(stack_list) == 1:
            return stack_list[0].histogram

        # Merge() requires a TList so fill that first
        tlist = ROOT.TList()
        for stack in stack_list[1:]:
            tlist.Add(stack.histogram)

        # Merge all histograms into first one
        stack_merge = stack_list[0].histogram
        stack_merge.Merge(tlist, ROOT.nullptr)

        return stack_merge

    @staticmethod
    def merge_efficiency_list(eff_list):
        """
        Leverages the builtin "+=" to merge all TEfficiency objects from the same process
        :param eff_list: Map of all TEfficiency objects
        :return: Merged TEfficiency object
        """
        if len(eff_list) < 1:
            return None
        elif len(eff_list) == 1:
            return eff_list[0].histogram

        merged_eff = eff_list[0].histogram
        for eff in eff_list[1:]:
            merged_eff.Add(eff.histogram)

        return merged_eff

    def plot_efficiency(self, xtotal, xpassed, idx):

        # Get the config to create the plot for this cut
        name, title, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]
        hist_total = ROOT.TH1D(name + "_total", title, bins, lower_lim, upper_lim)
        hist_passed = ROOT.TH1D(name + "_passed", title, bins, lower_lim, upper_lim)

        # If this is ndim array flatten it
        xtotal = ak.flatten(xtotal, axis=None)
        xpassed = ak.flatten(xpassed, axis=None)

        # Just a loop in c++ which does hist.FillN() (nullptr sets weights = 1)
        # FillN() only likes double* (python float64) so if array is another type, cast it to float64
        if isinstance(xtotal, ak.Array):
            if ak.type(xtotal.layout).dtype != 'float64' or ak.type(xpassed.layout).dtype != 'float64':
                xtotal = ak.values_astype(xtotal, np.float64)
                xpassed = ak.values_astype(xpassed, np.float64)
            hist_total.FillN(len(xtotal), ak.to_numpy(xtotal), ROOT.nullptr)
            hist_passed.FillN(len(xpassed), ak.to_numpy(xpassed), ROOT.nullptr)
        elif isinstance(xtotal, np.ndarray):
            if xtotal.dtype != np.float64 or xpassed.type != np.float64:
                xtotal = xtotal.astype('float64')
                xpassed = xpassed.astype('float64')
            hist_total.FillN(len(xtotal), xtotal, ROOT.nullptr)
            hist_passed.FillN(len(xpassed), xpassed, ROOT.nullptr)
        else:
            print("Unknown array type!")

        efficiency = ROOT.TEfficiency(name + "_eff", title, bins, lower_lim, upper_lim)
        efficiency.SetTotalHistogram(hist_total, "")
        efficiency.SetPassedHistogram(hist_passed, "")

        # Store this hist in our master map as HistogramData class object
        self.hist_data.append(HistogramData("efficiency", name, efficiency))

    def plot_particles(self, x, idx, precut):
        c = ROOT.TCanvas()
        legend = utils.legend_init_right()

        # Get the config to create the plot for this cut
        name, title, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]
        if precut:
            name = "precut_" + name
            title = "PreCut-" + title
        else:
            name = "postcut_" + name
            title = "PostCut-" + title

        hist = ROOT.TH1D(name, title, bins, lower_lim, upper_lim)

        # If this is ndim array flatten it
        x = ak.flatten(x, axis=None)

        # Just a loop in c++ which does hist.FillN() (nullptr sets weights = 1)
        # FillN() only likes double* (python float64) so if array is another type, cast it to float64
        if isinstance(x, ak.Array):
            if ak.type(x.layout).dtype != 'float64':
                x = ak.values_astype(x, np.float64)
            hist.FillN(len(x), ak.to_numpy(x), ROOT.nullptr)
        elif isinstance(x, np.ndarray):
            if x.dtype != np.float64:
                x = x.astype('float64')
            hist.FillN(len(x), x, ROOT.nullptr)
        else:
            print("Unknown array type!")

        legend.AddEntry(hist, name)
        hist.GetListOfFunctions().Add(legend)

        # Store this hist in our master map as HistogramData class object
        self.hist_data.append(HistogramData("hist", name, hist))

    def plot_particles_stack(self, x, x_pdg, idx, precut):
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
        name, title, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]
        if precut:
            name = "precut_" + name
            title = "PreCut-" + title
        else:
            name = "postcut_" + name
            title = "PostCut-" + title

        # Flatten the array once
        x_flat = ak.flatten(x, axis=None)
        x_pdg_flat = ak.flatten(x_pdg, axis=None)

        for i, pdg in enumerate(self.config["stack_pdg_list"]):

            hstack = ROOT.TH1D(name + "_" + str(pdg), title, bins, lower_lim, upper_lim)

            # Before plotting we flatten from 2D array shape=(<num event>, <num daughters>) to
            # 1D array shape=(<num event>*<num daughters>)
            pdg_list = [-211, -13, -11, 11, 13, 22, 111, 211, 321, 2212]
            pdg_filtered_array = plotting_utils.daughter_by_pdg(x_flat, x_pdg_flat, pdg, pdg_list)

            # If array is not empty fill it, if it is empty still add the histogram to the
            # stack so we can correctly merge the stacks later. Otherwise the PDGs get mixed.
            if len(pdg_filtered_array) > 0:
                # Now use the hist standard fill function and add color
                hstack.FillN(len(pdg_filtered_array), pdg_filtered_array, ROOT.nullptr)

            utils.set_hist_colors(hstack, utils.colors.get(utils.pdg2string.get(pdg, "Other"), 1),
                                  utils.colors.get(utils.pdg2string.get(pdg, "Other"), 1))

            # Get the fraction of PDG
            pdg_fraction = round((100. * len(pdg_filtered_array) / len(x_flat)), 2)

            legend.AddEntry(hstack, utils.pdg2string.get(pdg, "other") + "  " + str(pdg_fraction) + "%")
            # Only add the legend to one histogram so we don't have duplicates in the THStack
            if i == 0:
                hstack.GetListOfFunctions().Add(legend)

            stack.Add(hstack)

        # Draw and tidy the THStack
        stack.Draw()
        stack.SetName(name)
        stack.SetTitle(title.split(";")[0])
        stack.GetXaxis().SetTitle(title.split(";")[1])
        stack.GetYaxis().SetTitle(title.split(";")[2])

        # Store this hist in our master map as a HistogramData class
        self.hist_data.append(HistogramData("stack", name, stack))

        return

    def plot_process(self, x, precut):
        c = ROOT.TCanvas()
        legend = utils.legend_init_right()

        # Get the config to create the plot for this cut
        name, title, bins, lower_lim, upper_lim = list(self.hist_config[0].values())[0]
        if precut:
            name = "precut_proc_" + name
            title = "PreCut-Proc-" + title
        else:
            name = "postcut_proc_" + name
            title = "PostCut-Proc-" + title

        hist = ROOT.TH1I(name, title, len(self.true_process_list)+1, 0, len(self.true_process_list)+1)

        for proc in self.true_process_list:
            ROOT.fill_hist_th1i_string(proc, hist, np.int(ak.count_nonzero(x[proc])))

        legend.AddEntry(hist, name)
        hist.GetListOfFunctions().Add(legend)

        # Store this hist in our master map as HistogramData class object
        self.hist_data.append(HistogramData("hist", name, hist))

    def plot_process_stack(self, x, idx, variable, precut):
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

        # The name and binning should be the same for all particles
        name, title, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]
        if precut:
            name = "precut_proc_" + name
            title = "PreCut-" + title
        else:
            name = "postcut_proc_" + name
            title = "PostCut-" + title

        for i, proc in enumerate(self.true_process_list):

            # Filter and flatten the array
            proc_mask = x[proc]
            xprox = x[proc_mask]
            x_flat = ak.flatten(xprox[variable], axis=None)

            hstack = ROOT.TH1D(name + "_" + proc, title, bins, lower_lim, upper_lim)

            # Just a loop in c++ which does hist.FillN() (nullptr sets weights = 1)
            # FillN() only likes double* (python float64) so if array is another type, cast it to float64
            if len(x_flat) > 0:
                if isinstance(x_flat, ak.Array):
                    if ak.type(x_flat.layout).dtype != 'float64':
                        x_flat = ak.values_astype(x_flat, np.float64)
                    hstack.FillN(len(x_flat), ak.to_numpy(x_flat), ROOT.nullptr)
                elif isinstance(x_flat, np.ndarray):
                    if x_flat.dtype != np.float64:
                        x_flat = x_flat.astype('float64')
                    hstack.FillN(len(x_flat), x_flat, ROOT.nullptr)
                else:
                    print("Unknown array type!")

            utils.set_hist_colors(hstack, utils.proc_colors.get(proc, 1), utils.proc_colors.get(proc, 1))

            # Get the fraction of PDG
            pdg_fraction = round((100. * len(x_flat) / len(proc_mask)), 2)

            legend.AddEntry(hstack, proc + "  " + str(pdg_fraction) + "%")
            # Only add the legend to one histogram so we don't have duplicates in the THStack
            if i == 0:
                hstack.GetListOfFunctions().Add(legend)

            stack.Add(hstack)

        # Draw and tidy the THStack
        stack.Draw()
        stack.SetName(name)
        stack.SetTitle(title.split(";")[0])
        stack.GetXaxis().SetTitle(title.split(";")[1])
        stack.GetYaxis().SetTitle(title.split(";")[2])

        # Store this hist in our master map as a HistogramData class
        self.hist_data.append(HistogramData("stack", name, stack))

        return
