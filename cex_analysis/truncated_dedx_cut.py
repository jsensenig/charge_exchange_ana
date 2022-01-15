from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np


class TruncatedDedxCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "TruncatedDedxCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        track_score_mask = events["reco_daughter_PFP_trackScore_collection"] > 0.35

        good_dedx_mask = (events[cut_variable] > 0.) & (events[cut_variable] < 1000.)
        events["daughter_masked_dedx"] = ak.mean(events[cut_variable][good_dedx_mask], axis=2, mask_identity=False)

        # Plot the variable before making cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # We want to _reject_ events if there are daughter michel electrons (presumably from pions decays)
        # so negate the selection mask

        # Take the logical OR of each daughter in the events
        daughter_pion_mask = (events["daughter_masked_dedx"] > 0.5) & (events["daughter_masked_dedx"] < 2.8)
        daughter_pion_mask = daughter_pion_mask & track_score_mask

        selected_mask = np.any(daughter_pion_mask, axis=1)

        # Take the logical NOT of the array and cast it back to an Awkward array.
        # Casting into a Numpy array converts None to False (the negation then turns it True)
        selected_mask = ak.Array(~ak.to_numpy(selected_mask).data)

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
        doc_string = "Reject events which have daughter charged pions." \
                     "Since our signal is pi+ --> pi0 + N  (N = some number of nucleons)"
        return doc_string
