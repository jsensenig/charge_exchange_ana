from cex_analysis.event_selection_base import EventSelectionBase
from itertools import product
import awkward as ak
import numpy as np


class MichelDaughterCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "MichelDaughterCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Optimization rules
        self.opt_dict = {"cnn_michel_cut_param": [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]}
        self.optimization_values = product(self.opt_dict["cnn_michel_cut_param"])

        self.num_optimizations = 0
        if self.opt_dict:
            self.num_optimizations = np.cumprod([len(opt) for opt in self.opt_dict.values()])[-1]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]
        self.is_mc = self.config["is_mc"]

    def selection(self, events, hists, optimizing=False):
        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        daughter_michel_mask = events[cut_variable] > self.local_config["cnn_michel_cut_param"]

        # We want to _reject_ events if there are daughter michel electrons (presumably from pions decays)
        # so negate the selection mask

        # Take the logical OR of each daughter in the events
        selected_mask = np.any(daughter_michel_mask, axis=1)

        # Take the logical NOT of the array and cast it back to an Awkward array.
        # Casting into a Numpy array converts None to False (the negation then turns it True)
        #selected_mask = ak.Array(~ak.to_numpy(selected_mask).data)
        selected_mask = ak.to_numpy(selected_mask)
        selected_mask = ~selected_mask

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
                hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def cut_optimization(self):
        # get cut values and iterate to the next
        values = self.optimization_values.__next__()
        print("VALUES", values)
        self.local_config["cnn_michel_cut_param"] = values[0]

    def get_cut_doc(self):
        doc_string = "Reject events which have daughter charged pions." \
                     "Since our signal is pi+ --> pi0 + N  (N = some number of nucleons)"
        return doc_string
