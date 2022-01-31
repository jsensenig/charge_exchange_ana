from cex_analysis.event_selection_base import EventSelectionBase
import numpy as np


class TOFCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "TOFCut"
        self.config = config
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]

    def selection(self, events, hists, optimizing=False):

        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_daughter_pdf], precut=True, hists=hists)

        # Perform the actual cut on TOF
        # also cut out positrons since they are vetoed in the data
        #selected_mask = (self.local_config["lower"] < events[cut_variable][:, 0]) & \
        #                (events[cut_variable][:, 0] < self.local_config["upper"]) & \
        #                (events["true_beam_PDG"] != -11)

        selected_mask = (events[cut_variable][:, 0] < 97.) & (events["true_beam_PDG"] != -11)

        print("TOF Selected Events True Beam PDG: ", np.unique(events["true_beam_PDG", selected_mask], return_counts=True))
        print("TOF Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0", selected_mask], return_counts=True))

        # Plot the variable before after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_daughter_pdf, selected_mask],
                                     precut=False, hists=hists)

        # Plot the efficiency
        if not optimizing:
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

    def cut_optimization(self):
        pass

    def get_cut_doc(self):
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
