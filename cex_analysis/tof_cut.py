from cex_analysis.event_selection_base import EventSelectionBase


class TOFCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "TOFCut"
        self.local_config = self.config[self.cut_name]
        self.cut_variable = self.local_config["cut_variable"]
        self.reco_daughter_pdf = self.config["reco_daughter_pdg"]

    def selection(self, events):

        # Plot the variable before making cut
        self.plot_particles_base(events=events[self.cut_variable], pdg=events[self.reco_daughter_pdf], precut=True)

        # Perform the actual cut on the TOF
        selected_mask = (self.local_config["lower"] < events[self.cut_variable]) & \
                        (events[self.cut_variable] < self.local_config["upper"])

        # FIXME accept an array
        #super().efficiency(self.cut_name, selected_mask, events["tof"])

        # Plot the variable before after cut
        self.plot_particles_base(events=events[self.cut_variable, selected_mask],
                                 pdg=events[self.reco_daughter_pdf, selected_mask],
                                 precut=False)

        # Return event selection mask
        return selected_mask

    def plot_particles_base(self, events, pdg, precut):
        super().plot_particles_stack(x=events, x_pdg=pdg, cut=self.cut_name, precut=precut)

    def get_cut_doc(self):
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
