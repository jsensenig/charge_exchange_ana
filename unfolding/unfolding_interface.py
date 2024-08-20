from abc import abstractmethod
import json
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

import cross_section_utils as xsec_utils
from cex_analysis.true_process import TrueProcess
from cex_analysis.plot_utils import string2code
from bethe_bloch_utils import BetheBloch
from systematics.corrections import *
from systematics.systematics import *
from cex_analysis.plot_utils import bin_width_np, bin_centers_np


class XSecVariablesBase:

    def __init__(self, config, is_training):
        self.is_training = is_training
        self.xsec_vars = {}

        self.correction_classes = get_all_corrections()
        self.syst_classes = get_all_systematics()
        self.corrections = {}
        self.systematics = {}

        self.true_process = TrueProcess()

        self.apply_correction = config["apply_correction"]
        self.correction_list = config["correction_list"]
        self.apply_systematic = config["apply_systematic"]
        self.systematic_list = config["systematic_list"]

        # Load the systematic and corrections
        self.load_syst_classes(config=config, apply_corrections=self.apply_correction,
                               apply_systematics=self.apply_systematic)

    @abstractmethod
    def get_xsec_variable(self, event_record, reco_mask, apply_cuts=True):
        """
        Get the specified cross-section's variable(s).
        """
        pass
    
    def apply_corrections_and_systematics(self, events):

        if self.apply_correction:
            for corr in self.correction_list:
                correction_var = self.corrections[corr].get_correction_variable()
                self.xsec_vars[corr] = self.corrections[corr].apply(to_correct=events[correction_var])

        if self.apply_systematic:
            for syst in self.systematic_list:
                syst_var = self.systematics[syst].get_systematic_variable()
                self.systematics[syst].apply(syst_var=events[syst_var])

    def load_syst_classes(self, config, apply_corrections, apply_systematics):

        if apply_corrections:
            for corr in self.correction_classes:
                self.corrections[corr] = self.correction_classes[corr](config=config["correction_config"])

        if apply_systematics:
            for syst in self.syst_classes:
                self.corrections[syst] = self.correction_classes[syst](config=config["systematic_config"])

    def get_event_process(self, events, proc_list_name):

        event_int_proc = np.zeros(len(events))

        if proc_list_name == "all":
            proc_list = self.true_process.get_process_list()
        elif proc_list_name == "simple":
            proc_list = self.true_process.get_process_list_simple()
        elif proc_list_name == "daughter":
            proc_list = self.true_process.get_daughter_bkgd_list()
        elif proc_list_name == "beam":
            proc_list = self.true_process.get_beam_particle_list()
        else:
            print("Unknown process set:", proc_list_name)
            raise ValueError

        for proc in proc_list:
            proc_mask = events[proc]
            event_int_proc[proc_mask] = string2code[proc]

        return event_int_proc

    @staticmethod
    def configure(config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        with open(config_file, "r") as cfg:
            config = json.load(cfg)

        return config


class BeamPionVariables(XSecVariablesBase):
    def __init__(self, config_file, is_training, energy_slices):
        super().__init__(config=config_file, is_training=is_training)

        self.config = self.configure(config_file=config_file["interface_config"])
        self.signal_proc = self.config["signal_proc"]
        self.beam_pip_zlow, self.beam_pip_zhigh = self.config["beam_pip_zlow"], self.config["beam_pip_zhigh"]
        self.pip_mass = self.config["pip_mass"] # pip_mass = 0.13957039  # pi+/- [GeV/c]
        self.eslice_bin_array = energy_slices #self.config["eslice_bin_edges"] # FIXME inherit this from xsec

        self.beam_energy = self.config["beam_energy"]

        self.bethe_bloch = BetheBloch(mass=139.57, charge=1)

        # This is the dict that will hold the data being unfolded and used to calculate cross-section.
        # This is a very small subset of the original data
        self.xsec_vars = {}

    def get_xsec_variable(self, event_record, reco_int_mask, apply_cuts=True):
        self.xsec_vars["true_complete_slice_mask"] = np.ones(len(event_record)).astype(bool) if self.is_training else None
        self.xsec_vars["reco_complete_slice_mask"] = np.ones(len(event_record)).astype(bool)

        # Classify events
        if self.is_training:
            self.xsec_vars["beam_all_process"] = self.get_event_process(events=event_record, proc_list_name='all')
            self.xsec_vars["beam_simple_process"] = self.get_event_process(events=event_record, proc_list_name='simple')
            self.xsec_vars["beam_beam_process"] = self.get_event_process(events=event_record, proc_list_name='beam')
            self.apply_corrections_and_systematics(events=event_record)

        # Calculate and add the requisite columns to the event record
        # Make masks for incomplete initial and through-going pions
        true_up, true_down, reco_up, reco_down = self.beam_fiducial_volume(event_record=event_record)
        self.xsec_vars["true_upstream_mask"], self.xsec_vars["true_downstream_mask"] = true_up, true_down
        self.xsec_vars["reco_upstream_mask"], self.xsec_vars["reco_downstream_mask"] = reco_up, reco_down

        # Make true beam initial and end energy
        self.xsec_vars["true_beam_initial_energy"], self.xsec_vars["true_beam_end_energy"] = None, None
        if self.is_training:
            self.xsec_vars["true_beam_initial_energy"], self.xsec_vars["true_beam_end_energy"] = (
                self.make_true_beam_energy(event_record=event_record))

        # Make reco beam initial and end energy
        self.xsec_vars["reco_beam_initial_energy"], self.xsec_vars["reco_beam_end_energy"] = (
            self.make_reco_beam_energy(event_record=event_record))

        self.xsec_vars["true_beam_sig_int_energy"], _, self.xsec_vars["reco_beam_sig_int_energy"] = (
            self.make_beam_interacting(event_record=event_record, reco_int_mask=reco_int_mask))

        if self.is_training:
            self.xsec_vars["true_beam_sig_int_energy"][self.xsec_vars["true_downstream_mask"]] = -1.
            self.xsec_vars["true_beam_alt_sig_int_energy"][self.xsec_vars["true_downstream_mask"]] = -1.

        self.xsec_vars["reco_beam_sig_int_energy"][self.xsec_vars["reco_downstream_mask"]] = -1.

        # Mask out incomplete slices
        self.incomplete_energy_slice()

        # Now set all 3 histograms to -1 for events with incomplete slices
        if self.is_training:
            self.xsec_vars["true_beam_initial_energy"][~self.xsec_vars["true_complete_slice_mask"]] = -1.
            self.xsec_vars["true_beam_end_energy"][~self.xsec_vars["true_complete_slice_mask"]] = -1.
            self.xsec_vars["true_beam_sig_int_energy"][~self.xsec_vars["true_complete_slice_mask"]] = -1.

        self.xsec_vars["reco_beam_initial_energy"][~self.xsec_vars["reco_complete_slice_mask"]] = -1.
        self.xsec_vars["reco_beam_end_energy"][~self.xsec_vars["reco_complete_slice_mask"]] = -1.
        self.xsec_vars["reco_beam_sig_int_energy"][~self.xsec_vars["reco_complete_slice_mask"]] = -1.

        # Add an end Z position
        if self.is_training:
            self.xsec_vars["true_beam_endz"] = event_record["true_beam_traj_Z"][:, -1]

        self.xsec_vars["reco_beam_endz"] = np.ones(len(event_record)) * -999
        empty_mask = ak.count(event_record["reco_beam_calo_Z"], axis=1) > 0
        self.xsec_vars["reco_beam_endz"][empty_mask] = event_record["reco_beam_calo_Z"][empty_mask][:, -1]

        # Apply mask to events
        if self.is_training:
            true_mask = ~self.xsec_vars["true_upstream_mask"] & ~self.xsec_vars["reco_upstream_mask"]
        reco_mask = ~self.xsec_vars["reco_upstream_mask"] #true_mask if self.is_training else ~self.xsec_vars["reco_upstream_mask"]
        #reco_mask = true_mask if self.is_training else ~self.xsec_vars["reco_upstream_mask"]

        self.xsec_vars["full_len_true_mask"] = true_mask if self.is_training else None
        self.xsec_vars["full_len_reco_mask"] = reco_mask

        if not apply_cuts:
            return self.xsec_vars

        for k in self.xsec_vars:
            if k.split('_')[0] == 'true' and self.is_training:
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
            fiducial_zcut_mask = event_record["reco_beam_calo_Z", evt][1:] <= self.beam_pip_zhigh
            if np.count_nonzero(fiducial_zcut_mask) < 1:
                reco_beam_new_init_energy.append(-1)
                reco_beam_end_energy.append(-1)
                continue
            inc_energy = self.bethe_bloch.ke_along_track(self.xsec_vars["reco_ff_energy"][evt], ak.to_numpy(event_record["reco_track_cumlen", evt]))
            new_inc = xsec_utils.make_true_incident_energies(event_record["reco_beam_calo_Z", evt][1:][fiducial_zcut_mask], inc_energy[fiducial_zcut_mask])
            reco_beam_new_init_energy.append(new_inc[0])
            reco_beam_end_energy.append(new_inc[-1])
        return np.asarray(reco_beam_new_init_energy), np.asarray(reco_beam_end_energy)

    def make_true_beam_energy(self, event_record):
        """
        Get the incident energies for the slices
        Use the Bethe Bloch formula to calculate the KE loss as function of track length
        Makes true: incident and end energy
        """
        # Make true beam initial energy
        max_pt = ak.max(event_record["true_beam_traj_Z"][event_record["true_beam_traj_Z"] < self.beam_pip_zlow], axis=1)
        ff_mask = event_record["true_beam_traj_Z"] == max_pt
        true_initial_energy = ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:, 0])

        # The last KE point(s) is 0 by definition, mask these out so we can get the length to the point where it's still moving
        nz_ke_mask = event_record["true_beam_traj_KE"] > 0
        traj_dr = np.sqrt(np.square(event_record["true_beam_traj_X"][nz_ke_mask][:, 1:] - event_record["true_beam_traj_X"][nz_ke_mask][:, :-1])
                          + np.square(event_record["true_beam_traj_Y"][nz_ke_mask][:, 1:] - event_record["true_beam_traj_Y"][nz_ke_mask][:, :-1])
                          + np.square(event_record["true_beam_traj_Z"][nz_ke_mask][:, 1:] - event_record["true_beam_traj_Z"][nz_ke_mask][:, :-1]))

        true_len = ak.to_numpy(np.sum(traj_dr, axis=1))

        true_beam_end_energy = []
        for evt in range(len(event_record)):
            if len(event_record["true_beam_traj_Z"][evt]) < 1:
                true_beam_end_energy.append(-1)
                continue
            if event_record["true_beam_traj_Z", evt][-1] <= self.beam_pip_zlow:
                true_beam_end_energy.append(-1)
                continue
            if event_record["true_beam_traj_Z", evt][-1] <= self.beam_pip_zhigh:
                true_track_len = true_len[evt][-1]
            else:
                pts_in_fiducial_mask = event_record["true_beam_traj_Z", evt] < self.beam_pip_zhigh
                idx_in_fiducial = ak.argmax(event_record["true_beam_traj_Z", evt][pts_in_fiducial_mask])

                delta_len = true_len[idx_in_fiducial + 1] - true_len[idx_in_fiducial]
                delta_z = event_record["true_beam_traj_Z", evt][idx_in_fiducial + 1] - event_record["true_beam_traj_Z", evt][idx_in_fiducial]

                z_frac_in_fiducial = (self.beam_pip_zhigh - event_record["true_beam_traj_Z", evt][idx_in_fiducial]) / delta_z
                true_track_len = true_len[idx_in_fiducial] + delta_len * z_frac_in_fiducial

            end_energy = self.bethe_bloch.ke_at_length(true_initial_energy[evt], true_track_len)
            true_beam_end_energy.append(end_energy)

        return true_initial_energy, np.asarray(true_beam_end_energy)

    def make_reco_beam_energy(self, event_record):
        """
        Get the incident energies for the slices
        Use the Bethe Bloch formula to calculate the KE loss as function of track length
        Makes true: incident and end energy
        """
        # The last KE point(s) is 0 by definition, mask these out so we can get the length to the point where it's still moving
        traj_dr = np.sqrt(np.square(event_record["reco_beam_calo_X"][:, 1:] - event_record["reco_beam_calo_X"][:, :-1])
                          + np.square(event_record["reco_beam_calo_Y"][:, 1:] - event_record["reco_beam_calo_Y"][:, :-1])
                          + np.square(event_record["reco_beam_calo_Z"][:, 1:] - event_record["reco_beam_calo_Z"][:, :-1]))

        reco_len = ak.to_numpy(np.sum(traj_dr, axis=1))

        # Find the index where the beam particle crosses into the fiducial volume
        start_fid_idx = ak.argmax(event_record["reco_beam_calo_Z"][event_record["reco_beam_calo_Z"] < self.beam_pip_zlow], axis=1)

        # Front face of fiducial volume KE
        energy_smear = 0.
        if self.beam_energy == 2 and self.is_training:
            energy_smear = 1. + np.random.normal(0,0.1,len(event_record["beam_inst_P"]))

        reco_beam_energy = ak.to_numpy(event_record["beam_inst_P"] + energy_smear) - self.xsec_vars["UpstreamEnergyLoss"]
        reco_ff_energy = np.sqrt(np.square(self.pip_mass) + np.square(reco_beam_energy * 1.e3)) - self.pip_mass

        reco_beam_initial_energy = []
        reco_beam_end_energy = []
        for evt in range(len(event_record)):
            if len(event_record["reco_beam_calo_Z"][evt]) < 1:
                reco_beam_end_energy.append(-1)
                reco_beam_initial_energy.append(-1)
                continue
            if event_record["reco_beam_calo_Z", evt][-1] <= self.beam_pip_zlow:
                reco_beam_end_energy.append(-1)
                reco_beam_initial_energy.append(-999)
                continue

            initial_energy = self.bethe_bloch.ke_at_length(reco_ff_energy, reco_len[start_fid_idx])
            reco_beam_initial_energy.append(initial_energy)

            if event_record["reco_beam_calo_Z", evt][-1] <= self.beam_pip_zhigh:
                reco_track_len = reco_len[evt][-1]
            else:
                pts_in_fiducial_mask = event_record["reco_beam_calo_Z", evt] < self.beam_pip_zhigh
                idx_in_fiducial = ak.argmax(event_record["reco_beam_calo_Z", evt][pts_in_fiducial_mask])

                delta_len = reco_len[idx_in_fiducial + 1] - reco_len[idx_in_fiducial]
                delta_z = event_record["reco_beam_calo_Z", evt][idx_in_fiducial + 1] - event_record["reco_beam_calo_Z", evt][idx_in_fiducial]

                z_frac_in_fiducial = (self.beam_pip_zhigh - event_record["reco_beam_calo_Z", evt][idx_in_fiducial]) / delta_z
                reco_track_len = reco_len[idx_in_fiducial] + delta_len * z_frac_in_fiducial

            end_energy = self.bethe_bloch.ke_at_length(self.xsec_vars["reco_beam_initial_energy"][evt], reco_track_len)
            reco_beam_end_energy.append(end_energy)

        return np.asarray(reco_beam_initial_energy), np.asarray(reco_beam_end_energy)

    def beam_fiducial_volume(self, event_record):
        """
        Check whether an event is within the fiducial volume
        """
        true_up, true_down = None, None
        if self.is_training:
            true_up = ak.to_numpy(event_record["true_beam_traj_Z"][:, -1]) < self.beam_pip_zlow
            true_down = ak.to_numpy(event_record["true_beam_traj_Z"][:, -1]) > self.beam_pip_zhigh

        nz_mask = ak.count(event_record["reco_beam_calo_Z"], axis=1) > 0 # filter out zero lengths
        reco_up = np.ones(len(event_record)).astype(bool)    # no z-pts defaults to true
        reco_down = np.zeros(len(event_record)).astype(bool) # no z-pts defaults to false

        reco_up[nz_mask] = event_record["reco_beam_calo_Z"][nz_mask][:, -1] < self.beam_pip_zlow
        reco_down[nz_mask] = event_record["reco_beam_calo_Z"][nz_mask][:, -1] > self.beam_pip_zhigh

        return true_up, true_down, reco_up, reco_down

    def make_true_beam_initial_energy(self, event_record):
        """
        double ff_energy_reco = beam_inst_KE*1000 - Eloss;//12.74;
        double initialE_reco = bb.KEAtLength(ff_energy_reco, trackLenAccum[0]);
        """
        true_initial_energy = None
        if self.is_training:
            max_pt = ak.max(event_record["true_beam_traj_Z"][event_record["true_beam_traj_Z"] < self.beam_pip_zlow], axis=1)
            ff_mask = event_record["true_beam_traj_Z"] == max_pt
            true_initial_energy = ak.to_numpy(event_record["true_beam_traj_KE"][ff_mask][:, 0])

        return true_initial_energy

    def make_reco_beam_initial_energy(self, event_record):
        # Note the beam momentum is converted GeV -> MeV
        # beam inst sim wrong, 2GeV = 1Gev so shift it by 1 for 2GeV and 0 for 1GeV
        energy_smear = 0.
        if self.beam_energy == 2 and self.is_training:
            energy_smear = 1. + np.random.normal(0,0.1,len(event_record["beam_inst_P"]))

        reco_ff_energy = np.sqrt(np.square(self.pip_mass) + np.square(ak.to_numpy(event_record["beam_inst_P"] + energy_smear)*1.e3)) \
                       - self.pip_mass        
        us_eloss_mom = self.xsec_vars["UpstreamEnergyLoss"]
        reco_ff_energy -= np.sqrt(us_eloss_mom * us_eloss_mom + self.pip_mass*self.pip_mass) - self.pip_mass

        nz_mask = ak.to_numpy(ak.count(event_record["reco_track_cumlen"], axis=1) > 0)
        reco_initial_energy = np.ones(len(event_record)).astype('d') * -1.
        reco_initial_energy[nz_mask] = self.bethe_bloch.ke_at_point(reco_ff_energy[nz_mask],
                                                                    ak.to_numpy(event_record["reco_track_cumlen"][nz_mask][:, 0]))

        self.xsec_vars["reco_ff_energy"] = reco_ff_energy

        return reco_initial_energy

    def incomplete_energy_slice(self):

        if self.is_training:
            init_bin_idx = np.digitize(self.xsec_vars["true_beam_initial_energy"], bins=self.eslice_bin_array)
            end_bin_idx = np.digitize(self.xsec_vars["true_beam_end_energy"], bins=self.eslice_bin_array)
            alt_end_bin_idx = np.digitize(self.xsec_vars["true_beam_alt_end_energy"], bins=self.eslice_bin_array)
            self.xsec_vars["true_complete_slice_mask"] &= (init_bin_idx > end_bin_idx)
            self.xsec_vars["true_alt_complete_slice_mask"] = (init_bin_idx > alt_end_bin_idx)
            #overflow_mask = self.xsec_vars["true_beam_initial_energy"] < self.eslice_bin_array[-1]
            #self.xsec_vars["true_beam_initial_energy"][overflow_mask] -= bin_width_np(self.eslice_bin_array)
            #self.xsec_vars["true_complete_slice_mask"] &= overflow_mask

            self.xsec_vars["true_beam_initial_energy"] -= bin_width_np(self.eslice_bin_array)
            self.xsec_vars["true_beam_initial_energy"] = np.clip(self.xsec_vars["true_beam_initial_energy"], a_min=-1e3,
                                                                 a_max=self.eslice_bin_array[-1]-1) # make sure its in the bin if upper edge is not inclusive

        init_bin_idx = np.digitize(self.xsec_vars["reco_beam_initial_energy"], bins=self.eslice_bin_array)
        end_bin_idx = np.digitize(self.xsec_vars["reco_beam_end_energy"], bins=self.eslice_bin_array)

        self.xsec_vars["reco_complete_slice_mask"] &= (init_bin_idx > end_bin_idx)

        # FIXME I think it should be initial_E -= <the nearest lower bin edge>
        # for histogram it doesn't matter as the current code ensures the initial_E ends up in the right bin
        # but the value of initial_E will be wrong by 0 < Delta_initial_E < Eslice width
        self.xsec_vars["reco_beam_initial_energy"] -= bin_width_np(self.eslice_bin_array)
        self.xsec_vars["reco_beam_initial_energy"] = np.clip(self.xsec_vars["reco_beam_initial_energy"], a_min=-1e3,
                                                             a_max=self.eslice_bin_array[-1]-1)

    def make_beam_interacting(self, event_record, reco_int_mask):
        """
        N_end: *Any* pi+ that interacts within the fiducial volume
        N_int: The signal definition interaction energy, CeX or all pion inelastic
        For each event,
        N_end = -1 if incomplete first eslice
        N_int = -1 if past fiducial volume z_high < end_ke OR incomplete first eslice
        """
        true_int = None
        alt_true_int = None
        if self.is_training:
            # Exclusive interaction
            true_int_mask = ak.to_numpy(event_record[self.signal_proc])
            true_int = np.ones(len(event_record)) * -1.
            alt_true_int = np.ones(len(event_record)) * -1.
            true_int[true_int_mask] = self.xsec_vars["true_beam_end_energy"][true_int_mask].copy()
            alt_true_int[true_int_mask] = self.xsec_vars["true_beam_alt_end_energy"][true_int_mask].copy()

        # Exclusive interactions
        # For interacting just apply mask to end KE to extract the interacting
        reco_int = np.ones(len(event_record)) * -1.
        reco_int[reco_int_mask] = self.xsec_vars["reco_beam_end_energy"][reco_int_mask]

        return true_int, alt_true_int, reco_int

    def plot_beam_vars(self, unfold_hist, err_ax0, err_ax1, err_ax2, bin_array, h1_limits, h2_limits, h3_limits, plot_reco=True):

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        h1, bx1 , _ = ax1.hist(self.xsec_vars["true_beam_initial_energy"], bins=bin_array[0],
                               edgecolor='black', label='True')
        if plot_reco: ax1.hist(self.xsec_vars["reco_beam_initial_energy"], bins=bin_array[0], alpha=0.8,
                               color='indianred',edgecolor='black', label='Reco')
        h2, bx2, _ = ax2.hist(self.xsec_vars["true_beam_end_energy"], bins=bin_array[1], edgecolor='black', label='True')
        if plot_reco: ax2.hist(self.xsec_vars["reco_beam_end_energy"], bins=bin_array[1], alpha=0.8,
                               color='indianred', edgecolor='black', label='Reco')
        h3, bx3, _ = ax3.hist(self.xsec_vars["true_beam_sig_int_energy"], bins=bin_array[2],
                              edgecolor='black', label='True')
        if plot_reco: ax3.hist(self.xsec_vars["reco_beam_sig_int_energy"], bins=bin_array[2], alpha=0.8,
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
    def __init__(self, config_file, is_training, energy_slices=None):
        super().__init__(config=config_file, is_training=is_training)

        self.config = self.configure(config_file=config_file["interface_config"])
        self.signal_proc = self.config["signal_proc"]

    def get_xsec_variable(self, event_record, reco_int_mask, apply_cuts=True):

        # Classify events
        if self.is_training:
            self.xsec_vars["pi0_all_process"] = self.get_event_process(events=event_record, proc_list_name='all')
            self.xsec_vars["pi0_simple_process"] = self.get_event_process(events=event_record, proc_list_name='simple')
            self.xsec_vars["pi0_daughter_process"] = self.get_event_process(events=event_record, proc_list_name='daughter')
            self.apply_corrections_and_systematics(events=event_record)

        true_pi0_energy, reco_pi0_energy = self.make_pi0_energy(event_record=event_record, reco_mask=reco_int_mask)
        self.xsec_vars["true_pi0_energy"] = true_pi0_energy
        self.xsec_vars["reco_pi0_energy"] = reco_pi0_energy

        true_cos_theta, reco_cos_theta = self.make_pi0_cos_theta(event_record=event_record, reco_mask=reco_int_mask)
        self.xsec_vars["true_pi0_cos_theta"] = true_cos_theta
        self.xsec_vars["reco_pi0_cos_theta"] = reco_cos_theta

        return self.xsec_vars

    def make_pi0_energy(self, event_record, reco_mask):
        true_pi0_energy = None
        if self.is_training:
            true_mask = event_record[self.signal_proc]
            true_pi0_energy = ak.to_numpy(np.sum(event_record["true_beam_Pi0_decay_startP"][true_mask], axis=1) * 1.e3) - 135.
            self.xsec_vars["true_gamma_energy"] = event_record["true_beam_Pi0_decay_startP"][true_mask]

        reco_pi0_energy = ak.to_numpy(event_record["fit_pi0_energy"][reco_mask]) - 135.0

        return true_pi0_energy, reco_pi0_energy

    def make_pi0_cos_theta(self, event_record, reco_mask):

        true_cos_theta = None
        if self.is_training:
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
            true_cos_theta = np.diag(beam_dir_unit @ full_len_daughter_dir.T)[true_mask]

        reco_cos_theta = ak.to_numpy(event_record["fit_pi0_cos_theta"][reco_mask])

        return true_cos_theta, reco_cos_theta

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
