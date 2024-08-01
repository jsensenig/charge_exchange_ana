from cex_analysis.event_selection_base import EventSelectionBase
import cross_section_utils as xsec_utils
from bethe_bloch_utils import BetheBloch
import awkward as ak
import numpy as np


class Pi0CalibCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "Pi0CalibCut"
        self.config = config
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]

        self.pip_mass = self.local_config["pip_mass"]
        self.bethe_bloch = BetheBloch(mass=self.pip_mass, charge=1)

    def get_beam_end_ke(self, events):
        """
        Get the incident energies for the slices
        Use the Bethe Bloch formula to calculate the KE loss as function of track length
        Makes reco: incident and end energy
        """
        beame = 1.  # beam inst sim wrong, 2GeV = 1Gev so shift it by 1 for 2GeV and 0 for 1GeV
        reco_ff_energy = np.sqrt(np.square(self.pip_mass) + np.square(ak.to_numpy(events["beam_inst_P"] + beame) * 1.e3)) \
                         - self.pip_mass

        reco_beam_end_energy = np.ones(len(events)) * -1.
        for evt in range(len(events)):
            if len(events["reco_beam_calo_Z"][evt]) < 1:
                continue
            inc_energy = self.bethe_bloch.ke_along_track(reco_ff_energy[evt], ak.to_numpy(events["reco_track_cumlen", evt]))
            reco_beam_end_energy[evt] = xsec_utils.make_true_incident_energies(events["reco_beam_calo_Z", evt], inc_energy)[-1]

        return reco_beam_end_energy

    def get_daughter_total_ke(self, events):
        dedx = events["reco_daughter_allTrack_calibrated_dEdX_SCE"]
        dedx_mask = (dedx > self.local_config["dedx_lower"]) & (dedx < self.local_config["dedx_upper"])
        # For each event: Sum over dE/dx for each particle axis=2 and then over all particles axis=1
        daughter_sum_ke = ak.sum(ak.sum(dedx[dedx_mask], axis=2), axis=1)

        return daughter_sum_ke

    def selection(self, events, hists, optimizing=False):

        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # Get the beam particle end KE
        events["reco_beam_end_ke"] = self.get_beam_end_ke(events=events)
        events["reco_daughter_total_ke"] = self.get_daughter_total_ke(events=events)

        # Difference between beam interaction KE and all outgoing (daughter) KE
        events["reco_delta_beam_daughter_ke"] = events["reco_beam_end_ke"] - events["reco_daughter_total_ke"]

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_daughter_pdf], precut=True, hists=hists)

        # Perform the cut on the beam particle endpoint
        beam_end_ke_mask = events["reco_beam_end_ke"] > self.local_config["reco_beam_ke_lower"] # > 400
        delta_ke_mask = np.abs(events["reco_delta_beam_daughter_ke"]) < self.local_config["delta_ke_limit"] # < 200

        ts_cut = self.local_config["track_score_cut"] # 0.5

        # ==0, >0, <3
        selected_mask = \
         (ak.count_nonzero(events["reco_daughter_PFP_trackScore_collection"] > ts_cut,
                           axis=1) == self.local_config["no_track_cut"]) & \
         (ak.count_nonzero(events["reco_daughter_PFP_trackScore_collection"] < ts_cut,
                           axis=1) > self.local_config["shower_count_low"]) & \
         (ak.count_nonzero(events["reco_daughter_PFP_trackScore_collection"] < ts_cut,
                           axis=1) < self.local_config["shower_count_high"])

        # Plot the variable before after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_daughter_pdf, selected_mask],
                                     precut=False, hists=hists)
            # Plot the efficiency
            self.efficiency(total_events=events, passed_events=events[selected_mask], cut=self.cut_name, hists=hists)

        # Return event selection mask
        return selected_mask & beam_end_ke_mask & delta_ke_mask

    def plot_particles_base(self, events, pdg, precut, hists):
        # hists.plot_process(x=events, precut=precut)
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            hists.plot_particles_stack(x=events[plot], x_pdg=events, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def cut_optimization(self):
        pass

    def get_cut_doc(self):
        doc_string = """ 
                        Cut on difference between beam interacting KE and KE of all outgoing particles.  
                        To select events with T_pi+ ~ T_pi0 for calibration of pi0 reconstruction. 
                     """
        return doc_string
