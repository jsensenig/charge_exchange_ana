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
        # Create a mask if the max shower energy is greater than the threshold
        max_energy_cut = max_energy > self.local_config["max_energy_cut"]
        # Reduce daughter level to event level mask
        # --> I think np.max already reduces by a dimension ie from daughter up to event level
        return max_energy_cut #np.any(max_energy_cut, axis=0)

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut"]

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events[cut_variable], pdg=events[self.reco_beam_pdg],
                                 precut=True, hists=hists)

        # Max shower energy mask to select only events with at least one large shower
        selected_mask = self.max_shower_energy_cut(events)

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

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
