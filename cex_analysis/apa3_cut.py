from cex_analysis.event_selection_base import EventSelectionBase
import threading
import json


class APA3Cut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "APA3Cut"
        self.config = config
        self.local_config = None
        self.local_hist_config = None
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.configure()

    def selection(self, events, hists):

        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events[cut_variable], pdg=events[self.reco_daughter_pdf],
                                 precut=True, hists=hists)

        # Perform the actual cut on TOF
        selected_mask = (self.local_config["lower"] < events[cut_variable]) & \
                        (events[cut_variable] < self.local_config["upper"])

        # Plot the variable before after cut
        self.plot_particles_base(events=events[cut_variable, selected_mask],
                                 pdg=events[self.reco_daughter_pdf, selected_mask],
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
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
