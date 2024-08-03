from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np


class BeamQualityCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "BeamQualityCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]
        self.is_mc = self.config["is_mc"]

    def beam_to_tpc_cut(self, events):

        """
        1. Calculate the difference between beam particle and the average divided by its RMS for X, Y, Z, XY
        """

        if self.is_mc:
            beam_startx_mean = self.local_config["beam_startX_MC"]
            beam_starty_mean = self.local_config["beam_startY_MC"]
            beam_startz_mean = self.local_config["beam_startZ_MC"]
            beam_startx_sigma = self.local_config["beam_startX_rms_MC"]
            beam_starty_sigma = self.local_config["beam_startY_rms_MC"]
            beam_startz_sigma = self.local_config["beam_startZ_rms_MC"]
            beam_anglex_mean = self.local_config["beam_anglex_deg_mc"]
            beam_angley_mean = self.local_config["beam_angley_deg_mc"]
            beam_anglez_mean = self.local_config["beam_anglez_deg_mc"]
        else:
            beam_startx_mean = self.local_config["beam_startX_Data"]
            beam_starty_mean = self.local_config["beam_startY_Data"]
            beam_startz_mean = self.local_config["beam_startZ_Data"]
            beam_startx_sigma = self.local_config["beam_startX_rms_Data"]
            beam_starty_sigma = self.local_config["beam_startY_rms_Data"]
            beam_startz_sigma = self.local_config["beam_startZ_rms_Data"]
            beam_anglex_mean = self.local_config["beam_anglex_deg_data"]
            beam_angley_mean = self.local_config["beam_angley_deg_data"]
            beam_anglez_mean = self.local_config["beam_anglez_deg_data"]

        # Shift the start to the mean and normalize by the RMS for each dimension
        beam_dx = (events[self.local_config["beam_startX"]] - beam_startx_mean) / beam_startx_sigma
        beam_dy = (events[self.local_config["beam_startY"]] - beam_starty_mean) / beam_starty_sigma
        beam_dz = (events[self.local_config["beam_startZ"]] - beam_startz_mean) / beam_startz_sigma

        # Convert to numpy array with shape (2,N) where N is number of events
        beam_xy = np.vstack((ak.to_numpy(beam_dx), ak.to_numpy(beam_dy))).T
        # Get the length of the pairs in the xy plane
        #beam_dxy = np.linalg.norm(beam_xy, axis=1)
        beam_dxy = np.sqrt(np.sum(beam_xy*beam_xy, axis=1))

        """ 
        2. Calculate the dot product between beam particle direction and the average beam direction
        """

        # Now check the angle between MC and the beam direction
        pointx = events[self.local_config["beam_endX"]] - events[self.local_config["beam_startX"]]
        pointy = events[self.local_config["beam_endY"]] - events[self.local_config["beam_startY"]]
        pointz = events[self.local_config["beam_endZ"]] - events[self.local_config["beam_startZ"]]
        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        beam_dir = np.vstack((ak.to_numpy(pointx), ak.to_numpy(pointy), ak.to_numpy(pointz))).T
        # Normalize the direction vector
        #norm = np.linalg.norm(beam_dir, axis=1)
        norm = np.sqrt(np.sum(beam_dir * beam_dir, axis=1))
        beam_dir_unit = beam_dir / np.stack((norm, norm, norm), axis=1)

        # Define the MC direction vector
        mc_direction_unit = np.cos(np.radians(np.array([beam_anglex_mean, beam_angley_mean, beam_anglez_mean])))
        # ...and normalize it
        #mc_direction_unit = mc_direction_unit / np.linalg.norm(mc_direction_unit)
        mc_direction_unit = mc_direction_unit / np.sqrt(np.sum(mc_direction_unit * mc_direction_unit))

        # Create N copies of the MC unit vector so we can take dot product with the beam direction
        beam_dir_mc_unit = np.full_like(beam_dir_unit, mc_direction_unit)
        # Take dot product. Not sure if there is a better way, here we matrix multiply the directions and
        # get the diagonal of resulting matrix which is the dot product of the vectors

        # Takes too much memory to do all at once so calculate in chunks of 10k events
        # beam_dot = np.diag(beam_dir_unit @ beam_dir_mc_unit.T)

        beam_dot = np.empty(len(events))

        proc_steps = list(np.arange(0, len(events), 10000)) + [-1]
        prev_idx = 0
        for step in proc_steps:
            beam_dot[prev_idx:step] = np.diag(beam_dir_unit[prev_idx:step] @ beam_dir_mc_unit[prev_idx:step].T)
            prev_idx = step

        """ 
        3. Apply cuts to the calculations above and combine the resulting masks
        """

        # Now apply cuts to \Delta{x,y,z} / \sigma_{x,y,z} and dot product
        mask_dx = (beam_dx > self.local_config["beam_start_min_dx"]) & (beam_dx < self.local_config["beam_start_max_dx"])
        mask_dy = (beam_dy > self.local_config["beam_start_min_dy"]) & (beam_dy < self.local_config["beam_start_max_dy"])
        mask_dz = (beam_dz > self.local_config["beam_start_min_dz"]) & (beam_dz < self.local_config["beam_start_max_dz"])
        mask_dxy = (beam_dxy > self.local_config["beam_start_min_dxy"]) & (beam_dxy < self.local_config["beam_start_max_dxy"])
        mask_angle = (beam_dot > self.local_config["min_angle"]) & (beam_dot < self.local_config["max_angle"])

        # Combine the masks together for the final selection
        return mask_dz & mask_dxy & mask_angle
        #return mask_dx & mask_dy & mask_dz & mask_angle

    def selection(self, events, hists, optimizing=False):
        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # The beam quality cut is already a mask, 1 if passed 0 if not
        # also these are already at the event level so it's okay as is
        # The first is the old cut and second the updated one
        selected_mask_old = events[cut_variable]
        selected_mask = self.beam_to_tpc_cut(events)

        print("Selected new/old", np.sum(selected_mask), " ", np.sum(selected_mask_old))

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
        doc_string = "Cut on beamline TOF to select beam particles"
        return doc_string
