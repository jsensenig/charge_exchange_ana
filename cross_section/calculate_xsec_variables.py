import awkward as ak
import numpy as np
import cross_section_utils as xsec_utils


class XSecVariables:
    def __init__(self, config):
        self.config = config
        self.signal_proc = self.config["signal_proc"]
        self.is_mc = self.config["is_mc"]
        self.beam_pip_zlow, self.beam_pip_zhigh = self.config["beam_pip_zlow"], self.config["beam_pip_zhigh"]
        self.pip_mass = self.config["pip_mass"] # pip_mass = 0.13957039  # pi+/- [GeV/c]
        self.eslice_bin_array = self.config["eslice_bin_edges"]

    def get_beam_pion_vars(self, event_record, reco_mask):
        # Calculate and add the requisite columns to the event record
        event_record = self.beam_pion_fiducial_volume(event_record=event_record)
        event_record = self.initial_beam_energy(event_record=event_record)
        event_record = self.incomplete_energy_slice(event_record=event_record)

        self.make_beam_incident(event_record=event_record)
        self.make_beam_end_and_int_ke(event_record=event_record, reco_mask=reco_mask)

        # mask out the upstream intercacting events

    def make_beam_incident(self, event_record):
        """
        Get the incident energies for the slices
        """
        if self.is_mc:
            event_record["true_beam_inc_ke"] = event_record["true_beam_traj_incidentEnergies"]

        event_record["reco_beam_inc_ke"] = event_record["reco_beam_traj_incidentEnergies"]

        return event_record

    def beam_pion_fiducial_volume(self, event_record):
        """
        Check whether an event is within the fiducial volume
        """
        if self.is_mc:
            event_record["true_upstream_endz"] = event_record["true_beam_traj_Z"][:, -1] < self.beam_pip_zlow
            event_record["true_downstream_endz"] = event_record["true_beam_traj_Z"][:, -1] > self.beam_pip_zhigh

        event_record["reco_upstream_endz"] = event_record["reco_beam_calo_Z"][:, -1] < self.beam_pip_zlow
        event_record["reco_downstream_endz"] = event_record["reco_beam_calo_Z"][:, -1] > self.beam_pip_zhigh

        return event_record

    def initial_beam_energy(self, event_record):
        """
        double ff_energy_reco = beam_inst_KE*1000 - Eloss;//12.74;
        double initialE_reco = bb.KEAtLength(ff_energy_reco, trackLenAccum[0]);
        """
        if self.is_mc:
            event_record["true_beam_initial_ke"] = event_record["true_beam_traj_KE"][:, 0]

        beam_inst_ke = np.sqrt(np.square(self.pip_mass) + np.square(event_record["beam_inst_P"] + 1.)) - self.pip_mass
        beam_inst_ke *= 1.e3  # convert to [MeV]
        beam_inst_ke -= 12.74 # FIXME temporary Eloss

        event_record["reco_beam_initial_ke"] = beam_inst_ke #FIXME should be BetheBloch(beam_inst_ke, cumlen[0])

        return event_record

    def incomplete_energy_slice(self, event_record):

        if self.is_mc:
            bin_idx = np.digitize(event_record["true_beam_initial_ke"], bins=self.eslice_bin_array)
            upper_edge = self.eslice_bin_array[bin_idx]
            event_record["true_incomplete_slice"] = event_record["true_beam_interactingEnergy"] < upper_edge

        # 0th bin is underflow so the bin index will give us the idx+1 of the bin_array which is the upper edge
        bin_idx = np.digitize(event_record["reco_beam_initial_ke"], bins=self.eslice_bin_array)
        upper_edge = self.eslice_bin_array[bin_idx]

        # E_end < upper_bin_edge -> incomplete
        event_record["reco_incomplete_slice"] = event_record["reco_beam_interactingEnergy"] < upper_edge

        return event_record

    def make_beam_end_and_int_ke(self, event_record, reco_mask):
        """
        N_end: *Any* pi+ that interacts within the fiducial volume
        N_int: The signal definition interaction energy, CeX or all pion inelastic
        For each event,
        N_end = -1 if incomplete first eslice
        N_int = -1 if past fiducial volume z_high < end_ke OR incomplete first eslice
        """
        interacting_ke = np.ones(len(event_record)) * -1.

        if self.is_mc:
            signal_mask = event_record[self.signal_proc]
            true_end_ke = event_record["true_beam_interactingEnergy"]
            # All pion inelastic interactions
            event_record["true_beam_end_ke"] = true_end_ke
            # Exclusive interaction
            valid_end = ~event_record["true_downstream_endz"]
            interacting_ke[signal_mask & valid_end] = true_end_ke[signal_mask & valid_end]
            event_record["true_beam_int_ke"] = interacting_ke

        # All pion inelastic interactions / reco_incomplete_slice
        reco_end_ke = event_record["reco_beam_interactingEnergy"]
        # All pion inelastic interactions
        event_record["reco_beam_end_ke"] = reco_end_ke
        # Exclusive interactions
        interacting_ke[:] = -1.
        valid_end = ~event_record["reco_downstream_endz"]
        interacting_ke[reco_mask & valid_end] = reco_end_ke[reco_mask & valid_end]
        event_record["reco_beam_int_ke"] = interacting_ke

        return event_record

    def make_pi0_energy(self):
        pass

    def make_pi0_cos_theta(self):
        pass
