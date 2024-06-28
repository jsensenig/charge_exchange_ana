import cex_analysis.plot_utils as utils
from cex_analysis.histogram_data import Hist1d, HistStack, HistEff
# from cex_analysis.histogram_data import HistogramData
from cex_analysis.true_process import TrueProcess
# import ROOT
import awkward as ak
#import cex_analysis.plot_utils
import plotting_utils


class HistogramV2:
    def __init__(self, config):
        self.config = config
        self.hist_config = None

        # Create a list to hold all plot objects
        self.hist_data = []

        # Get the true process list
        # self.true_process_list = TrueProcess.get_process_list()
        self.true_process_list = TrueProcess.get_process_list_simple()

    def get_hist_map(self):
        return self.hist_data

    def configure_hists(self, config):
        self.hist_config = config

    @staticmethod
    def generate_name(hist_name, hist_type, precut):
        type_name = 'eff_' if hist_type == 'efficiency' else hist_type + '_'
        if precut:
            name = 'precut_' + type_name + hist_name
            hname = 'precut_' + hist_name
        else:
            name = 'postcut_' + type_name + hist_name
            hname = 'postcut_' + hist_name

        return name, hname

    @staticmethod
    def merge_hist_list(hist_list):
        # If no histograms, skip
        if len(hist_list) < 1:
            return None

        # Sum all histograms into first one
        # should work for any hist type with
        # overloaded __add__
        hsum = hist_list[0].histogram
        for hist in hist_list[1:]:
            hsum = hsum + hist.histogram

        return hsum

    def plot_efficiency(self, xtotal, xpassed, idx):

        # Get the config to create the plot for this cut
        name, xlabel, ylabel, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]

        # If this is ndim array flatten it
        xtotal = ak.flatten(xtotal, axis=None)
        xpassed = ak.flatten(xpassed, axis=None)

        # Fill the hists
        eff_hist = HistEff(num_bins=bins, bin_range=[lower_lim, upper_lim], xlabel=xlabel, ylabel=ylabel)
        eff_hist.fill_total(x=xtotal, legend=name, weights=None)
        eff_hist.fill_passed(x=xpassed, legend=name, weights=None)

        # self.hist_data.append(HistogramData("efficiency", name, eff_hist, precut=False))
        full_name, hist_name = self.generate_name(hist_name=name, hist_type='efficiency', precut=False)
        self.hist_data.append({'name': full_name, 'hist_name': hist_name, 'type': 'efficiency', 'hist': eff_hist})

    def plot_particles(self, x, idx, precut):

        # Get the config to create the plot for this cut
        name, xlabel, ylabel, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]

        # If this is ndim array flatten it
        x = ak.flatten(x, axis=None)

        # Fill the hists
        hist = Hist1d(num_bins=bins, bin_range=[lower_lim, upper_lim], xlabel=xlabel, ylabel=ylabel)
        hist.fill_hist(x=x, legend=name, weights=None)

        # self.hist_data.append(HistogramData("hist", name, hist, precut=precut))
        full_name, hist_name = self.generate_name(hist_name=name, hist_type='hist', precut=precut)
        self.hist_data.append({'name': full_name, 'hist_name': hist_name, 'type': 'hist', 'hist': hist})

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

        # The name and binning should be the same for all particles
        name, xlabel, ylabel, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]

        # Flatten the array once
        x_flat = ak.flatten(x, axis=None)
        x_pdg_flat = ak.flatten(x_pdg, axis=None)

        # Keep all the PDG selected arrays in here so we can sort and add to THStack by count
        pdg_filtered_dict = {}
        pdg_lengths_dict = {}

        for pdg in self.config["stack_pdg_list"]:
            # Before plotting we flatten from 2D array shape=(<num event>, <num daughters>) to
            # 1D array shape=(<num event>*<num daughters>)
            pdg_filtered_dict[pdg] = plotting_utils.daughter_by_pdg(x_flat, x_pdg_flat, pdg, self.config["stack_pdg_list"])
            pdg_lengths_dict[pdg] = len(pdg_filtered_dict[pdg])

        # sorted() sorts the values in descending order
        sorted_length = sorted(pdg_lengths_dict, key=pdg_lengths_dict.get, reverse=True)
        stack = HistStack(num_bins=bins, bin_range=[lower_lim, upper_lim], xlabel=xlabel, ylabel=ylabel)
        # legend_list
        for i, pdg in enumerate(sorted_length):
            # Get the fraction of PDG
            pdg_fraction = round((100. * len(pdg_filtered_dict[pdg]) / len(x_flat)), 2)
            legend = (str(pdg) + "  " + str(len(pdg_filtered_dict[pdg])) + "/" + str(len(x_flat)) +
                      " (" + str(pdg_fraction) + "%)")

            if len(pdg_filtered_dict[pdg]) > 0:
                stack.fill_stack(x=pdg_filtered_dict[pdg], legend=legend, weights=None)

        # Store this hist in our master map as a HistogramData class
        # self.hist_data.append(HistogramData("stack", name, stack, precut=precut))
        full_name, hist_name = self.generate_name(hist_name=name, hist_type='stack', precut=precut)
        self.hist_data.append({'name': full_name, 'hist_name': hist_name, 'type': 'stack', 'hist': stack})

        return

    def plot_process(self, x, precut):

        # Get the config to create the plot for this cut
        name, xlabel, ylabel, bins, lower_lim, upper_lim = list(self.hist_config[0].values())[0]

        # Fill the hists
        hist = Hist1d(num_bins=bins, bin_range=[lower_lim, upper_lim], xlabel=xlabel, ylabel=ylabel)
        hist.fill_hist(x=x, legend=name, weights=None)

        # self.hist_data.append(HistogramData("hist", name, hist, precut=precut))
        full_name, hist_name = self.generate_name(hist_name=name, hist_type='hist', precut=precut)
        self.hist_data.append({'name': full_name, 'hist_name': hist_name, 'type': 'hist', 'hist': hist})

    def plot_process_stack(self, x, idx, variable, precut):
        """
        Make stacked plot of a given variable with each stack corresponding to a PDG
        :param precut: Is this pre or Post cut plot
        :param x: Array of single variable either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param x_pdg: Array of PDG codes with either shape=(<num event>) OR shape=(<num event>, <num daughters>)
        :param cut: str Cut name
        :return:
        """
        # The name and binning should be the same for all particles
        name, xlabel, ylabel, bins, lower_lim, upper_lim = list(self.hist_config[idx].values())[0]

        # Invalid values default to -999 so mask out if it's less than -900
        valid_mask = ak.flatten(x[variable], axis=None) > -900.
        total_daughters = len(ak.flatten(x[variable], axis=None)[valid_mask])

        process_dict = {}
        process_lengths_dict = {}

        for proc in self.true_process_list:
            # Filter and flatten the array
            proc_mask = x[proc]
            xproc = x[proc_mask]
            process_dict[proc] = ak.flatten(xproc[variable], axis=None)
            process_lengths_dict[proc] = len(process_dict[proc])

        # sorted() sorts the values in descending order
        sorted_length = sorted(process_lengths_dict, key=process_lengths_dict.get, reverse=True)
        stack = HistStack(num_bins=bins, bin_range=[lower_lim, upper_lim], xlabel=xlabel, ylabel=ylabel)
        for i, proc in enumerate(sorted_length):
            # Get the fraction of PDG
            proc_fraction = round((100. * len(process_dict[proc][process_dict[proc] > -900.]) / total_daughters), 2)
            legend = (str(utils.string2code[proc]) + "  " + str(len(process_dict[proc][process_dict[proc] > -900.])) +
                      "/" + str(total_daughters) + " (" + str(proc_fraction) + "%)")

            stack.fill_stack(x=process_dict[proc], legend=legend, weights=None)

        # Store this hist in our master map as a HistogramData class
        # self.hist_data.append(HistogramData("stack", name, stack, precut=precut))
        full_name, hist_name = self.generate_name(hist_name=name, hist_type='stack', precut=precut)
        self.hist_data.append({'name': full_name, 'hist_name': hist_name, 'type': 'stack', 'hist': stack})

        return
