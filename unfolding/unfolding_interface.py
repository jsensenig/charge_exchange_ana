from abc import abstractmethod
import json
import awkward as ak
import numpy as np

import cross_section_utils as xsec_utils
from cross_section.xsec_calculation import XSecTotal
from bethe_bloch_utils import BetheBloch
from cex_analysis.plot_utils import bin_width_np, bin_centers_np


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


class BeamPionVariables(XSecVariablesBase):
    def __init__(self, config_file, is_mc):
        super().__init__(is_mc=is_mc)

        self.config = self.configure(config_file=config_file)
        self.signal_proc = self.config["signal_proc"]
        self.beam_pip_zlow, self.beam_pip_zhigh = self.config["beam_pip_zlow"], self.config["beam_pip_zhigh"]
        self.pip_mass = self.config["pip_mass"] # pip_mass = 0.13957039  # pi+/- [GeV/c]
        self.eslice_bin_array = self.config["eslice_bin_edges"] # FIXME inherit this from xsec

        self.bethe_bloch = BetheBloch(mass=139.57, charge=1)

        # This is the dict that will hold the data being unfolded and used to calculate cross-section.
        # This is a very small subset of the original data
        self.xsec_vars = {}

    def get_xsec_variable(self, event_record, reco_mask):
        self.xsec_vars["true_xsec_mask"] = np.ones(len(event_record)).astype(bool) if self.is_mc else None
        self.xsec_vars["reco_xsec_mask"] = np.ones(len(event_record)).astype(bool)

        # Calculate and add the requisite columns to the event record
        # Make masks for incomplete initial and through-going pions
        true_up, true_down, reco_up, reco_down = self.beam_pion_fiducial_volume(event_record=event_record)
        self.xsec_vars["true_upstream_mask"] = true_up
        self.xsec_vars["true_downstream_mask"] = true_down
        self.xsec_vars["reco_upstream_mask"] = reco_up
        self.xsec_vars["reco_downstream_mask"] = reco_down

        true_initial_energy, complete_slices, reco_initial_energy = self.initial_beam_energy(event_record=event_record)
        self.xsec_vars["true_beam_initial_energy"] = true_initial_energy
        self.xsec_vars["true_beam_end_energy"] = ak.to_numpy(event_record["true_beam_traj_KE"][:, -2])
        self.xsec_vars["reco_beam_initial_energy"] = reco_initial_energy
        self.xsec_vars["true_xsec_mask"] &= complete_slices

        if self.is_mc:
            self.xsec_vars["true_beam_new_init_energy"] = ak.to_numpy(event_record["true_beam_traj_KE"][:, 0])
            #self.xsec_vars["true_xsec_mask"] &= ((self.xsec_vars["true_beam_new_init_energy"] >= self.eslice_bin_array[0]) &
            #                                     (self.xsec_vars["true_beam_new_init_energy"] <= self.eslice_bin_array[-1]))
            #self.xsec_vars["true_xsec_mask"] &= ((self.xsec_vars["true_beam_end_energy"] >= self.eslice_bin_array[0]) &
            #                                     (self.xsec_vars["true_beam_end_energy"] <= self.eslice_bin_array[-1]))

        new_init_energy, new_end_energy = self.make_reco_beam_incident(event_record=event_record)

        # Mask out events which do not have intial or end energies within our range of interest
        #self.xsec_vars["reco_xsec_mask"] &= ((new_init_energy >= self.eslice_bin_array[0]) &
        #                                     (new_init_energy <= self.eslice_bin_array[-1]))
        #self.xsec_vars["reco_xsec_mask"] &= ((new_end_energy >= self.eslice_bin_array[0]) &
        #                                     (new_end_energy <= self.eslice_bin_array[-1]))

        # Add the reco incident
        self.xsec_vars["reco_beam_new_init_energy"] = new_init_energy
        self.xsec_vars["reco_beam_new_end_energy"] = new_end_energy

        true_all_int, true_int, reco_all_int, reco_int = self.make_beam_int_ke(event_record=event_record,
                                                                               reco_mask=reco_mask)

        if self.is_mc:
            self.xsec_vars["true_beam_all_int_energy"] = true_all_int
            self.xsec_vars["true_beam_sig_int_energy"] = true_int
            self.xsec_vars["true_beam_all_int_energy"][self.xsec_vars["true_downstream_mask"]] = -1.
            self.xsec_vars["true_beam_sig_int_energy"][self.xsec_vars["true_downstream_mask"]] = -1.

        self.xsec_vars["reco_beam_all_int_energy"] = reco_all_int
        self.xsec_vars["reco_beam_sig_int_energy"] = reco_int
        self.xsec_vars["reco_beam_all_int_energy"][self.xsec_vars["reco_downstream_mask"]] = -1.
        self.xsec_vars["reco_beam_sig_int_energy"][self.xsec_vars["reco_downstream_mask"]] = -1.

        # Mask out incomplete slices
        self.incomplete_energy_slice()

        # Add an end Z position
        if self.is_mc:
            self.xsec_vars["true_beam_endz"] = event_record["true_beam_traj_Z"][:, -1]

        self.xsec_vars["reco_beam_endz"] = np.ones(len(event_record)) * -999
        empty_mask = ak.count(event_record["reco_beam_calo_Z"], axis=1) > 0
        self.xsec_vars["reco_beam_endz"][empty_mask] = event_record["reco_beam_calo_Z"][empty_mask][:,-1]

        # Apply mask to events
        true_mask = ~self.xsec_vars["true_upstream_mask"] & self.xsec_vars["true_xsec_mask"]
        reco_mask = ~self.xsec_vars["reco_upstream_mask"] & self.xsec_vars["reco_xsec_mask"]

        for k in self.xsec_vars:
            if k.split('_')[0] == 'true':
                self.xsec_vars[k] = self.xsec_vars[k][true_mask]
            elif k.split('_')[0] == 'reco':
                self.xsec_vars[k] = self.xsec_vars[k][reco_mask]

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
            inc_energy = self.bethe_bloch.ke_along_track(self.xsec_vars["reco_ff_energy"][evt], ak.to_numpy(event_record["reco_track_cumlen", evt]))
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
        beame = 0. # beam inst sim wrong, 2GeV = 1Gev so shift it by 1 for 2GeV and 0 for 1GeV

        true_initial_energy = None
        if self.is_mc:
            #minimum = ak.min(abs(event_record["true_beam_traj_Z"] - 0.2), axis=1)
            #ff_mask = abs(event_record["true_beam_traj_Z"] - 0.2) == minimum
            #true_initial_energy = ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:, 0])
            #############
            max_pt = ak.max(event_record["true_beam_traj_Z"][event_record["true_beam_traj_Z"] < self.beam_pip_zlow], axis=1)
            ff_mask = event_record["true_beam_traj_Z"] == max_pt
            init = np.digitize(ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:,0]), bins=self.eslice_bin_array)
            end = np.digitize(ak.to_numpy(event_record["true_beam_traj_KE"][:, -2]), bins=self.eslice_bin_array)
            complete_slices = init != end
            # Shift down by bin width so its entered into the first full energy slice
            tmp_init_pts = ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:,0]) - bin_width_np(self.eslice_bin_array)
            true_initial_energy = np.clip(tmp_init_pts, a_min=-1e3, a_max=self.eslice_bin_array[-1])

        # Note the beam momentum is converted GeV -> MeV
        reco_ff_energy = np.sqrt(np.square(self.pip_mass) + np.square(ak.to_numpy(event_record["beam_inst_P"] + beame)*1.e3)) \
                       - self.pip_mass        
        reco_ff_energy -= 12.74 # FIXME temporary Eloss

        nz_mask = ak.to_numpy(ak.count(event_record["reco_track_cumlen"], axis=1) > 0)
        reco_initial_energy = np.ones(len(event_record)).astype('d') * -1.
        reco_initial_energy[nz_mask] = self.bethe_bloch.ke_at_point(reco_ff_energy[nz_mask], ak.to_numpy(event_record["reco_track_cumlen"][nz_mask][:, 0]))

        #self.xsec_vars["true_xsec_mask"] &= ((true_initial_energy >= self.eslice_bin_array[0]) &
        #                                     (true_initial_energy <= self.eslice_bin_array[-1]))
        #self.xsec_vars["reco_xsec_mask"] &= ((reco_initial_energy >= self.eslice_bin_array[0]) &
        #                                     (reco_initial_energy <= self.eslice_bin_array[-1]))

        self.xsec_vars["reco_ff_energy"] = reco_ff_energy

        return true_initial_energy, complete_slices, reco_initial_energy

    def incomplete_energy_slice(self):

        if self.is_mc:
            init_bin_idx = np.digitize(self.xsec_vars["true_beam_new_init_energy"], bins=self.eslice_bin_array)
            end_bin_idx = np.digitize(self.xsec_vars["true_beam_end_energy"], bins=self.eslice_bin_array)
            true_incomplete_slice = init_bin_idx == end_bin_idx
            #self.xsec_vars["true_xsec_mask"] &= ~true_incomplete_slice

        init_bin_idx = np.digitize(self.xsec_vars["reco_beam_initial_energy"], bins=self.eslice_bin_array)
        end_bin_idx = np.digitize(self.xsec_vars["reco_beam_new_end_energy"], bins=self.eslice_bin_array)

        self.xsec_vars["reco_xsec_mask"] &= (init_bin_idx != end_bin_idx)

        self.xsec_vars["reco_beam_initial_energy"] -= bin_width_np(self.eslice_bin_array)
        self.xsec_vars["reco_beam_initial_energy"] = np.clip(self.xsec_vars["reco_beam_initial_energy"], a_min=-1e3, a_max=self.eslice_bin_array[-1])

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
            true_all_int_mask = ~self.xsec_vars["true_upstream_mask"] #& ~self.xsec_vars["true_downstream_mask"]
            true_all_int = np.ones(len(event_record)) * -1.
            true_all_int[true_all_int_mask] = self.xsec_vars["true_beam_end_energy"][true_all_int_mask].copy()
            # Exclusive interaction
            true_int_mask = ak.to_numpy(event_record[self.signal_proc]) #& true_all_int_mask
            true_int = np.ones(len(event_record)) * -1.
            true_int[true_int_mask] = self.xsec_vars["true_beam_end_energy"][true_int_mask].copy()

        # All pion inelastic interactions
        reco_all_int_mask = ~self.xsec_vars["reco_upstream_mask"] & ~self.xsec_vars["reco_downstream_mask"]
        reco_all_int = np.ones(len(event_record)) * -1.
        reco_all_int[reco_all_int_mask] = self.xsec_vars["reco_beam_new_end_energy"][reco_all_int_mask]
        # Exclusive interactions
        # For interacting just apply mask to end KE to extract the interacting
        reco_int_mask = reco_mask #& reco_all_int_mask
        reco_int = np.ones(len(event_record)) * -1.
        reco_int[reco_int_mask] = self.xsec_vars["reco_beam_new_end_energy"][reco_int_mask]

        return true_all_int, true_int, reco_all_int, reco_int


class Pi0Variables(XSecVariablesBase):
    def __init__(self, config_file, is_mc):
        super().__init__(is_mc=is_mc)

        self.config = self.configure(config_file=config_file)
        self.signal_proc = self.config["signal_proc"]

    def get_xsec_variable(self, event_record, reco_mask):
        true_pi0_energy, reco_pi0_energy = self.make_pi0_energy(event_record=event_record, reco_mask=reco_mask)
        self.xsec_vars["true_pi0_energy"] = true_pi0_energy
        self.xsec_vars["reco_pi0_energy"] = reco_pi0_energy

        true_cos_theta, reco_cos_theta = self.make_pi0_cos_theta(event_record=event_record, reco_mask=reco_mask)
        self.xsec_vars["true_pi0_cos_theta"] = true_cos_theta
        self.xsec_vars["reco_pi0_cos_theta"] = reco_cos_theta

        return self.xsec_vars

    def make_pi0_energy(self, event_record, reco_mask):
        true_pi0_energy = None
        if self.is_mc:
            true_mask = event_record[self.signal_proc]
            true_pi0_energy = ak.to_numpy(np.sum(event_record["true_beam_Pi0_decay_startP"][true_mask], axis=1) * 1.e3)

        reco_pi0_energy = ak.to_numpy(np.sum(event_record["true_beam_Pi0_decay_startP"][reco_mask], axis=1) * 1.e3)

        return true_pi0_energy, reco_pi0_energy

    def make_pi0_cos_theta(self, event_record, reco_mask):
        true_mask = event_record[self.signal_proc]
        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector and normalize
        beam_dir = np.vstack((ak.to_numpy(event_record["true_beam_endPx"]),
                              ak.to_numpy(event_record["true_beam_endPy"]),
                              ak.to_numpy(event_record["true_beam_endPz"]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)
        beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)

        # Calculate the pi0 direction
        full_len_daughter_dir = np.zeros(shape=(len(event_record), 3))
        one_pi0_mask = ak.count_nonzero(event_record["true_beam_daughter_PDG"] == 111, axis=1) == 1

        pi0_daughter_mask = event_record["true_beam_daughter_PDG"][one_pi0_mask] == 111
        pi0_dir_px = ak.to_numpy(event_record["true_beam_daughter_startPx"][one_pi0_mask][pi0_daughter_mask])[:, 0]
        pi0_dir_py = ak.to_numpy(event_record["true_beam_daughter_startPy"][one_pi0_mask][pi0_daughter_mask])[:, 0]
        pi0_dir_pz = ak.to_numpy(event_record["true_beam_daughter_startPz"][one_pi0_mask][pi0_daughter_mask])[:, 0]

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        pi0_dir = np.vstack((pi0_dir_px, pi0_dir_py, pi0_dir_pz)).T
        pi0_norm = np.linalg.norm(pi0_dir, axis=1)
        pi0_dir_unit = pi0_dir / np.stack((pi0_norm, pi0_norm, pi0_norm), axis=1)

        full_len_daughter_dir[one_pi0_mask] = pi0_dir_unit

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        true_cos_theta = np.diag(beam_dir_unit @ full_len_daughter_dir.T)

        return true_cos_theta[true_mask], true_cos_theta[reco_mask]
