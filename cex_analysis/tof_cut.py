from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np


class TOFCut(EventSelectionBase):
    def __init__(self, config, cut_name):
        super().__init__(config)

        self.cut_name = cut_name
        self.config = config
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]
        self.is_mc = self.config["is_mc"]

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
        # select only events with valid beam instrumentatin data
        non_positron_mask = events["beam_inst_C0"] == 0
        valid_bi = events["beam_inst_valid"]
        
        valid_trigger_type_mask = (events["beam_inst_trigger"] == 12) 
        valid_num_momenta = events["beam_inst_nMomenta"] == 1
        valid_num_tracks = events["beam_inst_nTracks"] == 1
        valid_reco = events["reco_reconstructable_beam_event"] != 0
        
        valid_beam_particle = non_positron_mask & valid_bi & valid_trigger_type_mask & valid_num_momenta & valid_num_tracks & valid_reco

        selected_mask = np.zeros(len(events)).astype(bool)
        selected_mask[valid_beam_particle] = (self.local_config["lower"] < events[valid_beam_particle][cut_variable][:, 0]) & \
                        (events[valid_beam_particle][cut_variable][:, 0] < self.local_config["upper"])

        #                (events["true_beam_PDG"] != -11)
        # selected_mask = (events[cut_variable][:, 0] < 97.) & (events["true_beam_PDG"] != -11)

        if self.is_mc:
            print("TOF Selected Events True Beam PDG: ", np.unique(events["true_beam_PDG", selected_mask], return_counts=True))
            print("TOF Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0", selected_mask], return_counts=True))

        # Plot the variable before after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_daughter_pdf, selected_mask],
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
        pass

    def get_cut_doc(self):
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
