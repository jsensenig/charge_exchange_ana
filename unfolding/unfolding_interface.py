from abc import abstractmethod
import json
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

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
        true_up, true_down, reco_up, reco_down = self.beam_fiducial_volume(event_record=event_record)
        self.xsec_vars["true_upstream_mask"], self.xsec_vars["true_downstream_mask"] = true_up, true_down
        self.xsec_vars["reco_upstream_mask"], self.xsec_vars["reco_downstream_mask"] = reco_up, reco_down

        # Get initial beam energy
        true_initial_energy, reco_initial_energy = self.make_beam_initial_energy(event_record=event_record)
        self.xsec_vars["true_beam_initial_energy"] = true_initial_energy
        self.xsec_vars["reco_beam_initial_energy"] = reco_initial_energy

        new_init_energy, end_energy = self.make_reco_beam_incident(event_record=event_record)

        # Add the end energy
        if self.is_mc:
            self.xsec_vars["true_beam_end_energy"] = ak.to_numpy(event_record["true_beam_traj_KE"][:, -2])
        self.xsec_vars["reco_beam_end_energy"] = end_energy

        true_int, reco_int = self.make_beam_interacting(event_record=event_record, reco_mask=reco_mask)

        if self.is_mc:
            self.xsec_vars["true_beam_sig_int_energy"] = true_int
            self.xsec_vars["true_beam_sig_int_energy"][self.xsec_vars["true_downstream_mask"]] = -1.

        self.xsec_vars["reco_beam_sig_int_energy"] = reco_int
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

        self.xsec_vars["full_len_true_mask"] = true_mask
        self.xsec_vars["full_len_reco_mask"] = reco_mask

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
        reco_beam_end_energy = []
        for evt in range(len(event_record)):
            if len(event_record["reco_beam_calo_Z"][evt]) < 1:
                reco_beam_new_init_energy.append(-1)
                reco_beam_end_energy.append(-1)
                continue
            inc_energy = self.bethe_bloch.ke_along_track(self.xsec_vars["reco_ff_energy"][evt], ak.to_numpy(event_record["reco_track_cumlen", evt]))
            new_inc = xsec_utils.make_true_incident_energies(event_record["reco_beam_calo_Z", evt], inc_energy)
            reco_beam_new_init_energy.append(new_inc[0])
            reco_beam_end_energy.append(new_inc[-1])
        return np.asarray(reco_beam_new_init_energy), np.asarray(reco_beam_end_energy)

    def beam_fiducial_volume(self, event_record):
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

    def make_beam_initial_energy(self, event_record):
        """
        double ff_energy_reco = beam_inst_KE*1000 - Eloss;//12.74;
        double initialE_reco = bb.KEAtLength(ff_energy_reco, trackLenAccum[0]);
        """
        beame = 0. # beam inst sim wrong, 2GeV = 1Gev so shift it by 1 for 2GeV and 0 for 1GeV

        true_initial_energy = None
        if self.is_mc:
            max_pt = ak.max(event_record["true_beam_traj_Z"][event_record["true_beam_traj_Z"] < self.beam_pip_zlow], axis=1)
            ff_mask = event_record["true_beam_traj_Z"] == max_pt
            true_initial_energy = ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:, 0])

        # Note the beam momentum is converted GeV -> MeV
        reco_ff_energy = np.sqrt(np.square(self.pip_mass) + np.square(ak.to_numpy(event_record["beam_inst_P"] + beame)*1.e3)) \
                       - self.pip_mass        
        reco_ff_energy -= 12.74 # FIXME temporary Eloss

        nz_mask = ak.to_numpy(ak.count(event_record["reco_track_cumlen"], axis=1) > 0)
        reco_initial_energy = np.ones(len(event_record)).astype('d') * -1.
        reco_initial_energy[nz_mask] = self.bethe_bloch.ke_at_point(reco_ff_energy[nz_mask],
                                                                    ak.to_numpy(event_record["reco_track_cumlen"][nz_mask][:, 0]))

        self.xsec_vars["reco_ff_energy"] = reco_ff_energy

        return true_initial_energy, reco_initial_energy

    def incomplete_energy_slice(self):

        if self.is_mc:
            init_bin_idx = np.digitize(self.xsec_vars["true_beam_initial_energy"], bins=self.eslice_bin_array)
            end_bin_idx = np.digitize(self.xsec_vars["true_beam_end_energy"], bins=self.eslice_bin_array)
            self.xsec_vars["true_xsec_mask"] &= (init_bin_idx != end_bin_idx)
            self.xsec_vars["true_beam_initial_energy"] -= bin_width_np(self.eslice_bin_array)
            self.xsec_vars["true_beam_initial_energy"] = np.clip(self.xsec_vars["true_beam_initial_energy"], a_min=-1e3,
                                                                 a_max=self.eslice_bin_array[-1])

        init_bin_idx = np.digitize(self.xsec_vars["reco_beam_initial_energy"], bins=self.eslice_bin_array)
        end_bin_idx = np.digitize(self.xsec_vars["reco_beam_end_energy"], bins=self.eslice_bin_array)

        self.xsec_vars["reco_xsec_mask"] &= (init_bin_idx != end_bin_idx)

        self.xsec_vars["reco_beam_initial_energy"] -= bin_width_np(self.eslice_bin_array)
        self.xsec_vars["reco_beam_initial_energy"] = np.clip(self.xsec_vars["reco_beam_initial_energy"], a_min=-1e3,
                                                             a_max=self.eslice_bin_array[-1])

    def make_beam_interacting(self, event_record, reco_mask):
        """
        N_end: *Any* pi+ that interacts within the fiducial volume
        N_int: The signal definition interaction energy, CeX or all pion inelastic
        For each event,
        N_end = -1 if incomplete first eslice
        N_int = -1 if past fiducial volume z_high < end_ke OR incomplete first eslice
        """
        true_int = None
        if self.is_mc:
            # Exclusive interaction
            true_int_mask = ak.to_numpy(event_record[self.signal_proc])
            true_int = np.ones(len(event_record)) * -1.
            true_int[true_int_mask] = self.xsec_vars["true_beam_end_energy"][true_int_mask].copy()

        # Exclusive interactions
        # For interacting just apply mask to end KE to extract the interacting
        reco_int_mask = reco_mask
        reco_int = np.ones(len(event_record)) * -1.
        reco_int[reco_int_mask] = self.xsec_vars["reco_beam_end_energy"][reco_int_mask]

        return true_int, reco_int

    def plot_beam_vars(self, unfold_hist, err_ax0, err_ax1, err_ax2, bin_array, h1_limits, h2_limits, h3_limits, plot_reco=True):
        true_mask = ~self.xsec_vars["true_upstream_mask"] & self.xsec_vars["true_xsec_mask"]
        reco_mask = ~self.xsec_vars["reco_upstream_mask"] & self.xsec_vars["reco_xsec_mask"]

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        h1, bx1 , _ = ax1.hist(self.xsec_vars["true_beam_initial_energy"][true_mask], bins=bin_array[0],
                               edgecolor='black', label='True')
        if plot_reco: ax1.hist(self.xsec_vars["reco_beam_initial_energy"][reco_mask], bins=bin_array[0], alpha=0.8,
                               color='indianred',edgecolor='black', label='Reco')
        h2, bx2, _ = ax2.hist(self.xsec_vars["true_beam_end_energy"][true_mask], bins=bin_array[1],
                              edgecolor='black', label='True')
        if plot_reco: ax2.hist(self.xsec_vars["reco_beam_end_energy"][reco_mask], bins=bin_array[1], alpha=0.8,
                               color='indianred', edgecolor='black', label='Reco')
        h3, bx3, _ = ax3.hist(self.xsec_vars["true_beam_sig_int_energy"][true_mask], bins=bin_array[2],
                              edgecolor='black', label='True')
        if plot_reco: ax3.hist(self.xsec_vars["reco_beam_sig_int_energy"][reco_mask], bins=bin_array[2], alpha=0.8,
                               color='indianred', edgecolor='black', label='Reco')

        ax1.errorbar(bin_centers_np(bx1), unfold_hist.sum(axis=2).sum(axis=1), err_ax0, bin_width_np(bx1[2:4]) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='Unfolded')
        ax2.errorbar(bin_centers_np(bx2), unfold_hist.sum(axis=2).sum(axis=0), err_ax1, bin_width_np(bx2[2:4]) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='Unfolded')
        ax3.errorbar(bin_centers_np(bx3), unfold_hist.sum(axis=1).sum(axis=0), err_ax2, bin_width_np(bx3[2:4]) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='Unfolded')
        ax1.set_title('$KE_{init}$', fontsize=14)
        ax2.set_title('$KE_{end}$', fontsize=14)
        ax3.set_title('$KE_{int}$', fontsize=14)
        ax1.set_xlim(h1_limits)
        ax2.set_xlim(h2_limits)
        ax3.set_xlim(h3_limits)
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax3.set_ylim(bottom=0, top=(np.max(h3[1:-1]) * 1.2))
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()

        print("True Init/End/Int", np.sum(h1), "/", np.sum(h2), "/", np.sum(h3))
        print("Reco Init/End/Int", unfold_hist.sum(axis=2).sum(axis=1).sum(), "/",
              unfold_hist.sum(axis=2).sum(axis=0).sum(), "/", unfold_hist.sum(axis=1).sum(axis=0).sum())


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
            true_pi0_energy = ak.to_numpy(np.sum(event_record["true_beam_Pi0_decay_startP"][true_mask], axis=1) * 1.e3) - 135.

        reco_pi0_energy = ak.to_numpy(np.sum(event_record["true_beam_Pi0_decay_startP"][reco_mask], axis=1) * 1.e3) - 135.

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

    def plot_pi0_vars(self, unfold_hist, err_ax0, err_ax1, bin_array, h1_limits, h2_limits, plot_reco=True):

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        h1, bx1, h1obj = ax1.hist(self.xsec_vars["true_pi0_energy"], bins=bin_array[0], edgecolor='black', label='True')
        rh1, _, _ = ax1.hist(self.xsec_vars["reco_pi0_energy"], bins=bin_array[0], alpha=0.8, color='indianred',
                             edgecolor='black', label='Reco')
        h2, bx2, _ = ax2.hist(self.xsec_vars["true_pi0_cos_theta"], bins=bin_array[1], edgecolor='black', label='True')
        rh2, _, _ = ax2.hist(self.xsec_vars["reco_pi0_cos_theta"], bins=bin_array[1], alpha=0.8, color='indianred',
                             edgecolor='black', label='Reco')

        ax1.errorbar(bin_centers_np(bx1), unfold_hist.sum(axis=1), err_ax0, bin_width_np(bx1[1:-1]) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='Unfolded')
        ax2.errorbar(bin_centers_np(bx2), unfold_hist.sum(axis=0), err_ax1, bin_width_np(bx2[1:-1]) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='Unfolded')
        ax1.set_title('$T_{\\pi^0}$', fontsize=16)
        ax2.set_title('$cos\\theta_{\\pi^0}$', fontsize=16)
        ax1.set_xlim(h1_limits)
        ax2.set_xlim(h2_limits)
        ax1.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        ax1.legend()
        ax2.legend()
        plt.show()

        print("True T_pi0/cos_pi0", np.sum(h1), "/", np.sum(h2))
        print("Reco T_pi0/cos_pi0", np.sum(rh1), "/", np.sum(rh2))
