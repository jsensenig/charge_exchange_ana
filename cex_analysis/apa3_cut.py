from cex_analysis.event_selection_base import EventSelectionBase
import awkward  as ak
import numpy as np


class APA3Cut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "APA3Cut"
        self.config = config
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]

    def beam_angle(self, events):
        # Now check the angle between beam start/end direction
        # 3rd element of each event uses the first/last 4points to get start/end direction
        start_dir_x = events["reco_beam_calo_startDirX"][:, 3]
        start_dir_y = events["reco_beam_calo_startDirY"][:, 3]
        start_dir_z = events["reco_beam_calo_startDirZ"][:, 3]
        end_dir_x = events["reco_beam_calo_endDirX"][:, 3]
        end_dir_y = events["reco_beam_calo_endDirY"][:, 3]
        end_dir_z = events["reco_beam_calo_endDirZ"][:, 3]

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        beam_start_dir = np.vstack((ak.to_numpy(start_dir_x), ak.to_numpy(start_dir_y), ak.to_numpy(start_dir_z))).T
        beam_end_dir = np.vstack((ak.to_numpy(end_dir_x), ak.to_numpy(end_dir_y), ak.to_numpy(end_dir_z))).T

        # Normalize the direction vector
        norm = np.linalg.norm(beam_start_dir, axis=1)
        beam_start_dir_unit = beam_start_dir / np.stack((norm, norm, norm), axis=1)

        norm = np.linalg.norm(beam_end_dir, axis=1)
        beam_end_dir_unit = beam_end_dir / np.stack((norm, norm, norm), axis=1)

        return np.diag(beam_start_dir_unit @ beam_end_dir_unit.T)

    def selection(self, events, hists, optimizing=False):

        # Add beam angle to events
        # Was going to use this to discriminate against pi+ QE events but didn't work.
        #events["beam_track_angle"] = self.beam_angle(events)

        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_daughter_pdf], precut=True, hists=hists)

        # Perform the cut on the beam particle endpoint
        selected_mask = (self.local_config["lower"] < events[cut_variable]) & \
                        (events[cut_variable] < self.local_config["upper"]) & \
                        (events["beam_track_angle"] > 0.2)

        # Plot the variable before after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_daughter_pdf, selected_mask],
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

    def cut_optimization(self):
        pass

    def get_cut_doc(self):
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
