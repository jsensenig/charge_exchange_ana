from abc import abstractmethod
import json
import awkward as ak
import numba as nb
import numpy as np

import cross_section_utils as xsec_utils
from bethe_bloch_utils import BetheBloch


class XSecVariablesBase:

    def __init__(self, is_mc):
        self.is_mc = is_mc
        self.xsec_vars = {}

    @abstractmethod
    def get_xsec_variable(self, event_record, reco_mask):
        """
        Get the specified cross-section's variable(s).
        """
        pass

    @nb.njit()
    def mask_list(self, key, mask):
        masked_list = []
        for el, m in zip(self.xsec_vars[key], mask):
            if not m: continue
            masked_list.append(el)

        return masked_list

    @staticmethod
    def configure(config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        with open(config_file, "r") as cfg:
            config = json.load(cfg)

        return config

# class BetheBloch(bethe_bloch_utils.BetheBloch):
#
#     def __init__(self, mass, charge):
#         super().__init__(mass, charge)


class BeamPionVariables(XSecVariablesBase):
    def __init__(self, config_file, is_mc):
        super().__init__(is_mc=is_mc)

        self.config = self.configure(config_file=config_file)
        self.signal_proc = self.config["signal_proc"]
        self.beam_pip_zlow, self.beam_pip_zhigh = self.config["beam_pip_zlow"], self.config["beam_pip_zhigh"]
        self.pip_mass = self.config["pip_mass"] # pip_mass = 0.13957039  # pi+/- [GeV/c]
        self.eslice_bin_array = self.config["eslice_bin_edges"]

        self.bethe_bloch = BetheBloch(mass=139.57, charge=1)

        # This is the dict that will hold the data being unfolded and used to calculate cross-section.
        # This is a very small subset of the original data
        self.xsec_vars = {}

    def get_xsec_variable(self, event_record, reco_mask):
        # Calculate and add the requisite columns to the event record
        # Make masks for incomplete initial and through-going pions
        true_up, true_down, reco_up, reco_down = self.beam_pion_fiducial_volume(event_record=event_record)
        self.xsec_vars["true_upstream_mask"] = true_up
        self.xsec_vars["true_downstream_mask"] = true_down
        self.xsec_vars["reco_upstream_mask"] = reco_up
        self.xsec_vars["reco_downstream_mask"] = reco_down

        true_initial_energy, reco_initial_energy = self.initial_beam_energy(event_record=event_record)
        self.xsec_vars["true_beam_initial_energy"] = true_initial_energy
        self.xsec_vars["reco_beam_initial_energy"] = reco_initial_energy

        if self.is_mc:
            self.xsec_vars["true_beam_new_init_energy"] = ak.to_numpy(event_record["true_beam_traj_KE"][:, 0])
            self.xsec_vars["true_beam_new_end_energy"] = ak.to_numpy(event_record["true_beam_interactingEnergy"])

        new_init_energy, new_end_energy = self.make_reco_beam_incident(event_record=event_record)

        # Add the reco incident
        self.xsec_vars["reco_beam_new_init_energy"] = new_init_energy
        self.xsec_vars["reco_beam_new_end_energy"] = new_end_energy

        true_all_int, true_int, reco_all_int, reco_int = self.make_beam_int_ke(event_record=event_record,
                                                                                                   reco_mask=reco_mask)

        if self.is_mc:
            self.xsec_vars["true_beam_all_int_energy"] = true_all_int
            self.xsec_vars["true_beam_sig_int_energy"] = true_int

        self.xsec_vars["reco_beam_all_int_energy"] = reco_all_int
        self.xsec_vars["reco_beam_sig_int_energy"] = reco_int

        #
        true_incomplete_slice, reco_incomplete_slice = self.incomplete_energy_slice()
        self.xsec_vars["true_incomplete_slice_mask"] = true_incomplete_slice
        self.xsec_vars["reco_incomplete_slice_mask"] = reco_incomplete_slice

        return self.xsec_vars

    def make_reco_beam_incident(self, event_record):
        """
        Get the incident energies for the slices
        Use the Bethe Bloch formula to calculate the KE loss as function of track length
        Makes reco: incident and end energy
        """
        reco_beam_new_init_energy = []
        reco_beam_new_end_energy = []
        for evt in range(len(event_record)):
            if len(event_record["reco_beam_calo_Z"][evt]) < 1:
                reco_beam_new_init_energy.append(-1)
                reco_beam_new_end_energy.append(-1)
                continue
            inc_energy = self.bethe_bloch.ke_along_track(self.xsec_vars["reco_beam_initial_energy"][evt], ak.to_numpy(event_record["reco_track_cumlen", evt]))
            new_inc = xsec_utils.make_true_incident_energies(event_record["reco_beam_calo_Z", evt], inc_energy)
            reco_beam_new_init_energy.append(new_inc[0])
            reco_beam_new_end_energy.append(new_inc[-1])
        return np.asarray(reco_beam_new_init_energy), np.asarray(reco_beam_new_end_energy)

    def beam_pion_fiducial_volume(self, event_record):
        """
        Check whether an event is within the fiducial volume
        """
        true_up, true_down = None, None
        if self.is_mc:
            true_up = ak.to_numpy(event_record["true_beam_traj_Z"][:, -1]) < self.beam_pip_zlow
            true_down = ak.to_numpy(event_record["true_beam_traj_Z"][:, -1]) > self.beam_pip_zhigh

        nz_mask = ak.count(event_record["reco_beam_calo_Z"], axis=1) > 0
        reco_up = np.ones(len(event_record)).astype(bool)    # no z-pts defaults to true
        reco_down = np.zeros(len(event_record)).astype(bool) # no z-pts defaults to false

        reco_up[nz_mask] = event_record["reco_beam_calo_Z"][nz_mask][:, -1] < self.beam_pip_zlow
        reco_down[nz_mask] = event_record["reco_beam_calo_Z"][nz_mask][:, -1] > self.beam_pip_zhigh

        return true_up, true_down, reco_up, reco_down

    def initial_beam_energy(self, event_record):
        """
        double ff_energy_reco = beam_inst_KE*1000 - Eloss;//12.74;
        double initialE_reco = bb.KEAtLength(ff_energy_reco, trackLenAccum[0]);
        """

        true_initial_energy = ak.to_numpy(event_record["true_beam_traj_KE"][:, 0]) if self.is_mc else None

        # Note the beam momentum is converted GeV -> MeV
        reco_initial_energy = np.sqrt(np.square(self.pip_mass) + np.square(ak.to_numpy(event_record["beam_inst_P"] + 1.)*1.e3)) \
                       - self.pip_mass        
        reco_initial_energy -= 12.74 # FIXME temporary Eloss

        return true_initial_energy, reco_initial_energy

    def incomplete_energy_slice(self):

        true_incomplete_slice = None
        if self.is_mc:
            bin_idx = np.digitize(self.xsec_vars["true_beam_new_init_energy"], bins=self.eslice_bin_array)
            for i, b in enumerate(self.eslice_bin_array):
                bin_idx[bin_idx == i] = b
            true_incomplete_slice = self.xsec_vars["true_beam_new_end_energy"] < bin_idx

        # 0th bin is underflow so the bin index will give us the idx+1 of the bin_array which is the upper edge
        bin_idx = np.digitize(self.xsec_vars["reco_beam_new_init_energy"], bins=self.eslice_bin_array)
        for i, b in enumerate(self.eslice_bin_array):
            bin_idx[bin_idx == i] = b

        # E_end < upper_bin_edge -> incomplete
        reco_incomplete_slice = self.xsec_vars["reco_beam_new_end_energy"] < bin_idx

        return true_incomplete_slice, reco_incomplete_slice

    def make_beam_int_ke(self, event_record, reco_mask):
        """
        N_end: *Any* pi+ that interacts within the fiducial volume
        N_int: The signal definition interaction energy, CeX or all pion inelastic
        For each event,
        N_end = -1 if incomplete first eslice
        N_int = -1 if past fiducial volume z_high < end_ke OR incomplete first eslice
        """
        true_all_int, true_int = None, None
        if self.is_mc:
            # All pion inelastic interactions
            true_all_int_mask = ~self.xsec_vars["true_upstream_mask"] & ~self.xsec_vars["true_downstream_mask"]
            true_all_int = np.ones(len(event_record)) * -1.
            true_all_int[true_all_int_mask] = self.xsec_vars["true_beam_new_end_energy"][true_all_int_mask]
            # Exclusive interaction
            true_int_mask = ak.to_numpy(event_record[self.signal_proc]) & true_all_int_mask
            true_int = np.ones(len(event_record)) * -1.
            true_all_int[true_int_mask] = self.xsec_vars["true_beam_new_end_energy"][true_int_mask]

        # All pion inelastic interactions
        reco_all_int_mask = ~self.xsec_vars["reco_upstream_mask"] & ~self.xsec_vars["reco_downstream_mask"]
        reco_all_int = np.ones(len(event_record)) * -1.
        reco_all_int[reco_all_int_mask] = self.xsec_vars["reco_beam_new_end_energy"][reco_all_int_mask]
        # Exclusive interactions
        # For interacting just apply mask to end KE to extract the interacting
        reco_int_mask = reco_mask & reco_all_int_mask
        reco_int = np.ones(len(event_record)) * -1.
        reco_int[reco_int_mask] = self.xsec_vars["reco_beam_new_end_energy"][reco_int_mask]

        return true_all_int, true_int, reco_all_int, reco_int


class Pi0Variables(XSecVariablesBase):
    def __init__(self, is_mc):
        super().__init__(is_mc=is_mc)

    def get_xsec_variable(self, event_record, reco_mask):
        pass

    def make_pi0_energy(self):
        pass

    def make_pi0_cos_theta(self):
        pass