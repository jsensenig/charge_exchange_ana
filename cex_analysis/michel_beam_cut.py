from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np


class MichelBeamCut(EventSelectionBase):
    def __init__(self, config, cut_name):
        super().__init__(config)

        self.cut_name = cut_name
        self.config = config
        self.reco_beam_pdg = self.config["reco_beam_pdg"]
        #self.chi2_ndof_var = "proton_chi2_ndof"

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]
        self.is_mc = self.config["is_mc"]

    def selection(self, events, hists, optimizing=False):
        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        events["beam_michel_score"] = np.where((events["reco_beam_vertex_nHits"] < 1), 0.,
                                              (events["reco_beam_vertex_michel_score"] / events["reco_beam_vertex_nHits"]))

        #events["beam_michel_score"] = (events["reco_beam_vertex_michel_score"] / events["reco_beam_vertex_nHits"])

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        selected_mask = events["beam_michel_score"] < self.local_config["cnn_michel_cut"]

        # We want to _reject_ events if there are daughter michel electrons (presumably from pions decays)
        # so negate the selection mask

        # Take the logical OR of each daughter in the events
        #selected_mask = np.any(beam_michel_mask, axis=0)

        # Take the logical NOT of the array and cast it back to an Awkward array.
        # Casting into a Numpy array converts None to False (the negation then turns it True)
        #numpy_selected_mask = ak.to_numpy(selected_mask).data
        #selected_mask = ak.Array(~numpy_selected_mask)

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
        pass

    def get_cut_doc(self):
        doc_string = "Reject events which have daughter charged pions." \
                     "Since our signal is pi+ --> pi0 + N  (N = some number of nucleons)"
        return doc_string
