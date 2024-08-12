from cex_analysis.event_selection_base import EventSelectionBase
import awkward  as ak
import numpy as np
from itertools import product


class Pi0NLLCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "Pi0NLLCut"
        self.config = config
        self.reco_daughter_pdf = self.config["reco_beam_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]
        self.is_mc = self.config["is_mc"]

        # Optimization rules
        self.opt_dict = {"nll_cut_param": [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
                         "inv_mass_cut_param": [50., 60., 70., 80., 90., 100., 110.]}
        self.optimization_values = product(self.opt_dict["nll_cut_param"], self.opt_dict["inv_mass_cut_param"])

        self.num_optimizations = 0
        if self.opt_dict:
            self.num_optimizations = np.cumprod([len(opt) for opt in self.opt_dict.values()])[-1]

    def dn_dalpha_distribution_mod(self, alpha, epi0):
        offset = 0.1
        min_angle = 2. * np.arcsin(135. / epi0)
        momentum = np.sqrt(epi0 ** 2 - 135. * 135.)
        beta = (momentum / 135.) * np.sqrt(1 / (1 + (momentum / 135.) ** 2))
        gamma = 1 / np.sqrt(1 - beta ** 2)

        diff_angle = 2 * (1 / (4. * gamma * beta)) * (np.cos(alpha / 2.) / np.sin(alpha / 2.) ** 2) * (
                1 / np.sqrt(gamma ** 2 * np.sin(alpha / 2.) ** 2 - 1))

        if alpha < (min_angle + np.radians(offset)):
            min_alpha = min_angle + np.radians(offset)
            trans_point = 2 * (1 / (4. * gamma * beta)) * (np.cos(min_alpha / 2.) / np.sin(min_alpha / 2.) ** 2) * (
                    1 / np.sqrt(gamma ** 2 * np.sin(min_alpha / 2.) ** 2 - 1))
            diff_angle = trans_point * np.exp(50. * (alpha - min_alpha))

        return diff_angle

    def invariant_mass(self, events):
        return np.sqrt(2. * events["fit_pi0_gamma_energy1"] * events["fit_pi0_gamma_energy2"] * (
                    1. - np.cos(np.radians(events["fit_pi0_gamma_oa"]))))

    def selection(self, events, hists, optimizing=False):

        events["fit_pi0_oa_nll"] = [-np.log(self.dn_dalpha_distribution_mod(alpha=np.radians(oa), epi0=energy) + 1.e-200)
                                    for energy, oa in zip(events["fit_pi0_energy"], events["fit_pi0_gamma_oa"])]

        events["pi0_invariant_mass"] = self.invariant_mass(events=events)

        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_daughter_pdf], precut=True, hists=hists)

        # Perform the cut on the beam particle endpoint
        selected_mask = (ak.to_numpy(events["fit_pi0_oa_nll"]) < self.local_config["nll_cut_param"]) & \
                        (ak.to_numpy(events["pi0_invariant_mass"]) > self.local_config["inv_mass_cut_param"])


        # Plot the variable before after cut
        if not optimizing:
            self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_daughter_pdf, selected_mask],
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
