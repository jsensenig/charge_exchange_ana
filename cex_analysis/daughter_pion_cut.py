from cex_analysis.event_selection_base import EventSelectionBase
import numpy as np


class DaughterPionCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "DaughterPionCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

    def chi2_ndof(self, events):
        return (events[self.local_config["proton_chi2"]] / events[self.local_config["proton_ndof"]]) \
               > self.local_config["chi2_ndof_cut"]

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events[cut_variable], pdg=events[self.reco_beam_pdg],
                                 precut=True, hists=hists)

        daughter_pion_mask = self.chi2_ndof(events)

        # Combine all event level masks
        # We want to _reject_ events if there are daughter charged pions so negate the selection mask
        selected_mask = ~np.any(daughter_pion_mask, axis=1)

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
        doc_string = "Reject events which have daughter charged pions." \
                     "Since our signal is pi+ --> pi0 + N  (N = some number of nucleons)"
        return doc_string
