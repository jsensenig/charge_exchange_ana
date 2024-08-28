from cex_analysis.event_selection_base import EventSelectionBase
from itertools import product
import awkward as ak
import numpy as np


class ShowerPreCut(EventSelectionBase):
    def __init__(self, config, cut_name):
        super().__init__(config)

        self.cut_name = cut_name
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Optimization rules
        self.opt_dict = {"cnn_shower_cut_param": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
                         "small_energy_shower_cut_param": [5., 10., 15., 20., 25., 30., 35., 40., 45., 50.]}
        self.optimization_values = product(self.opt_dict["cnn_shower_cut_param"], self.opt_dict["small_energy_shower_cut_param"])

        self.num_optimizations = 0
        if self.opt_dict:
            self.num_optimizations = np.cumprod([len(opt) for opt in self.opt_dict.values()])[-1]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]
        self.is_mc = self.config["is_mc"]

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut_param"]

    def min_shower_energy_cut(self, events):
        return events[self.local_config["shower_energy_var"]] > self.local_config["small_energy_shower_cut_param"]

    def shower_count_cut(self, events):
        """
        1. CNN cut to select shower-like daughters
        2. Energy cut, eliminate small showers from e.g. de-excitation gammas
        :param events:
        :return:
        """
        # Perform a 2 step cut on showers, get a daughter mask from each
        cnn_shower_mask = self.cnn_shower_cut(events)
        min_shower_energy = self.min_shower_energy_cut(events)
        nhit_mask = events["reco_daughter_PFP_nHits"] > 80.

        # Shower selection mask
        shower_mask = cnn_shower_mask & nhit_mask #&  min_shower_energy

        # We want to count the number of potential showers in each event
        shower_count = np.count_nonzero(events[self.local_config["shower_energy_var"], shower_mask], axis=1)

        if self.is_mc:
            shower_print_cex = shower_count[events["single_charge_exchange"]]
            print("sCEX Old Shower Count: 0/1/2/3/4/5 =",
                  ak.count_nonzero(shower_print_cex == 0), "/",
                  ak.sum(shower_print_cex == 1), "/",
                  ak.sum(shower_print_cex == 2), "/",
                  ak.sum(shower_print_cex == 3), "/",
                  ak.sum(shower_print_cex == 4), "/",
                  ak.sum(shower_print_cex == 5))

            shower_print_pi0_prod = shower_count[events["pi0_production"]]
            print("Pi0 Prod Old Shower Count: 0/1/2/3/4/5 =",
                  ak.count_nonzero(shower_print_pi0_prod == 0), "/",
                  ak.sum(shower_print_pi0_prod == 1), "/",
                  ak.sum(shower_print_pi0_prod == 2), "/",
                  ak.sum(shower_print_pi0_prod == 3), "/",
                  ak.sum(shower_print_pi0_prod == 4), "/",
                  ak.sum(shower_print_pi0_prod == 5))

        # Create the event mask, true if there are 2 candidate showers
        return (shower_count > 0) #& (shower_count < 3)

    def selection(self, events, hists, optimizing=False):

        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Pre-select events with at least one and no more than 2 showers
        selected_mask = self.shower_count_cut(events)

        # Unfortunately the number of Pandora showers introduces None's into the
        # array which can't be handled downstream. To remedy this we replace all
        # None's with 'True' (this is a bool array). We only want to use the the
        # Pandora selection to help remove impurities while avoiding it's inefficiency
        # therefore we replace None's (unknown) with True.
        #selected_mask = np.where(ak.is_none(selected_mask), True, selected_mask)

        # Plot the variable after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_beam_pdg, selected_mask],
                                     precut=False, hists=hists)
            # Plot the efficiency
            self.efficiency(total_events=events, passed_events=events[selected_mask], cut=self.cut_name, hists=hists)

        # Return event selection mask
        return selected_mask

    def plot_particles_base(self, events, pdg, precut, hists):
        # hists.plot_process(x=events, precut=precut)
        for idx, plot in enumerate(self.local_hist_config):
            if self.is_mc:
                hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            if list(plot.keys())[0] == "max_shower_energy":
                continue
            if self.is_mc:
                hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def cut_optimization(self):
        # get cut values and iterate to the next
        values = self.optimization_values.__next__()
        print("VALUES", values)
        self.local_config["cnn_shower_cut_param"] = values[0]
        self.local_config["small_energy_shower_cut_param"] = values[1]

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
