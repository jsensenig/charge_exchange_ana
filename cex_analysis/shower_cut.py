from cex_analysis.event_selection_base import EventSelectionBase
import numpy as np
import threading
import json


class ShowerCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "ShowerCut"
        self.config = config
        self.local_config = None
        self.local_hist_config = None
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.configure()

    def max_shower_energy_cut(self, events):
        # Get the maximum shower energy for each event
        max_energy = np.max(events[self.local_config["shower_energy_var"]], axis=1)
        max_energy_cut =  max_energy > self.local_config["max_energy_cut"]
        # Reduce daughter level to event level mask
        return np.any(max_energy_cut, axis=0)

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut"]

    def min_shower_energy_cut(self, events):
        return events[self.local_config["shower_energy_var"]] > self.local_config["small_energy_shower_cut"]

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

        # Shower selection mask
        shower_mask = cnn_shower_mask & min_shower_energy

        # We want to count the number of potential showers in each event
        shower_count = np.count_nonzero(events[self.local_config["shower_energy_var"], shower_mask], axis=1)

        return shower_count == 2

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events[cut_variable], pdg=events[self.reco_beam_pdg],
                                 precut=True, hists=hists)

        # Max shower
        max_shower_energy_mask = self.max_shower_energy_cut(events)

        # Candidate shower count
        shower_count_mask = self.shower_count_cut(events)

        # Combine all event level masks
        selected_mask = max_shower_energy_mask & shower_count_mask

        # Plot the variable after cut
        self.plot_particles_base(events=events[cut_variable, selected_mask],
                                 pdg=events[self.reco_beam_pdg, selected_mask],
                                 precut=False, hists=hists)

        # Plot the efficiency
        self.efficiency(total_events=events[cut_variable], passed_events=events[cut_variable, selected_mask],
                        cut=self.cut_name, hists=hists)

        # Return event selection mask
        return selected_mask

    def plot_particles_base(self, events, pdg, precut, hists):
        hists.plot_particles_stack(x=events, x_pdg=pdg, cut=self.cut_name, precut=precut)
        hists.plot_particles(x=events, cut=self.cut_name, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        hists.plot_efficiency(xtotal=total_events, xpassed=passed_events, cut=cut)

    def configure(self):
        config_file = self.config[self.cut_name]["config_file"]
        lock = threading.Lock()
        lock.acquire()
        with open(config_file, "r") as cfg:
            tmp_config = json.load(cfg)
            self.local_config = tmp_config[self.cut_name]
            self.local_hist_config = tmp_config["histograms"]
        lock.release()
        return

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string