from cex_analysis.event_selection_base import EventSelectionBase
import numpy as np


class MaxShowerEnergyCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "MaxShowerEnergyCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

    def max_shower_energy_cut(self, events):
        # We only want to look at shower-like daughters
        cnn_shower_mask = self.cnn_shower_cut(events)
        # Get the maximum daughter shower energy for each event
        max_energy = np.max(events[self.local_config["shower_energy_var"], cnn_shower_mask], axis=1)
        # Return a mask: true if the max shower energy is greater than the threshold
        return max_energy > self.local_config["max_energy_cut"]

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut"]

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Max shower energy mask to select only events with at least one large shower
        selected_mask = self.max_shower_energy_cut(events)

        # Plot the variable after cut
        self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_beam_pdg, selected_mask],
                                 precut=False, hists=hists)

        # Plot the efficiency
        self.efficiency(total_events=events, passed_events=events[selected_mask], cut=self.cut_name, hists=hists)

        # Return event selection mask
        return selected_mask

    def plot_particles_base(self, events, pdg, precut, hists):
        hists.plot_process(x=events, precut=precut)
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
