from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np
import tmp.shower_count_direction as sdir
import tmp.shower_likelihood as sll
from itertools import product


class ShowerCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "ShowerCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Optimization rules
        self.opt_dict = {"likelihood_l01_param": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1],
                         "likelihood_l12_param": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1]}
        self.optimization_values = product(self.opt_dict["likelihood_l01_param"], self.opt_dict["likelihood_l12_param"])

        self.num_optimizations = 0
        if self.opt_dict:
            self.num_optimizations = np.cumprod([len(opt) for opt in self.opt_dict.values()])[-1]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)
        self.optimize = self.local_config["optimize_cut"]

        # FIXME test shower counting, make local class object
        self.dir = sdir.ShowerDirection()
        self.ll = sll.ShowerLikelihood(likelihood01=self.local_config["likelihood_l10_param"],
                                       likelihood12=self.local_config["likelihood_l12_param"])

    def transform_point_to_spherical(self, events):
        # Column names
        tmp_x = "reco_daughter_PFP_shower_spacePts_tmpX"
        tmp_y = "reco_daughter_PFP_shower_spacePts_tmpY"
        tmp_z = "reco_daughter_PFP_shower_spacePts_tmpZ"

        # Make a temp column we can operate on
        events[tmp_x] = events["reco_daughter_PFP_shower_spacePts_X"]
        events[tmp_y] = events["reco_daughter_PFP_shower_spacePts_Y"]
        events[tmp_z] = events["reco_daughter_PFP_shower_spacePts_Z"]

        # Shift origin to beam interaction vertex
        events[tmp_x] = events[tmp_x] - events["reco_beam_endX"]
        events[tmp_y] = events[tmp_y] - events["reco_beam_endY"]
        events[tmp_z] = events[tmp_z] - events["reco_beam_endZ"]

        # rho
        xy = events[tmp_x] ** 2 + events[tmp_z] ** 2
        # R
        events["reco_daughter_PFP_shower_spacePts_R"] = np.sqrt(xy + events[tmp_z] ** 2)
        # Theta
        events["reco_daughter_PFP_shower_spacePts_Theta"] = np.arctan2(np.sqrt(xy), -events[tmp_y]) * (360. / (2 * np.pi))
        # Phi
        events["reco_daughter_PFP_shower_spacePts_Phi"] = np.arctan2(events[tmp_z],events[tmp_x]) * (360. / (2 * np.pi))

    def selection(self, events, hists, optimizing=False):
        # First we configure the histograms we want to make
        if not optimizing:
            hists.configure_hists(self.local_hist_config)

        # Plot the variable before making cut
        if not optimizing:
            self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Method which counts the peaks in theta-phi histogram as a proxy for
        # number of showers.
        peak_count_list = []
        cex_peak_count_list = []
        cex_npi0_list = []
        cex_nsp_list = []
        peak_angles = []
        shower_selection_mask = []
        for i in range(0, len(events)):
            if events[i] is None or len(events["reco_daughter_PFP_shower_spacePts_X", i]) < 50:
                if events["single_charge_exchange", i]:
                    cex_peak_count_list.append(0)
                    cex_npi0_list.append(0)
                shower_selection_mask.append(False)
                peak_count_list.append(0)
                peak_angles.append(np.array([]))
                continue
            coord = self.dir.transform_to_spherical(events=events[i])
            if coord is None:
                if events["single_charge_exchange", i]:
                    cex_peak_count_list.append(0)
                    cex_npi0_list.append(0)
                shower_selection_mask.append(False)
                peak_count_list.append(0)
                peak_angles.append(np.array([]))
                continue
            rmask = coord[:, 3] <= (3.5 * 14.)  # 3.5 * X_0
            shower_dir = []
            if np.count_nonzero(rmask) > 0:
                shower_dir = self.dir.get_shower_direction_unit(coord[rmask])
                # shower_dir = self.dir.get_shower_direction_unit(coord)
            peak_angles.append(shower_dir)
            peak_count = len(shower_dir)
            valid_energy_mask = events["reco_daughter_PFP_shower_spacePts_E", i] > 0.
            esum = ak.sum(events["reco_daughter_PFP_shower_spacePts_E", i][valid_energy_mask])
            pred_npi0 = self.ll.classify_npi0(coord[:, 3:5], esum, classify_2d=False)
            if pred_npi0:
                shower_selection_mask.append(True)
                if events["single_charge_exchange", i]:
                    cex_npi0_list.append(1)
                    cex_nsp_list.append(events["reco_daughter_PFP_shower_spacePts_count", i])
            else:
                shower_selection_mask.append(False)
                if events["single_charge_exchange", i]:
                    cex_npi0_list.append(pred_npi0)
            peak_count_list.append(len(shower_dir))
            if events["single_charge_exchange", i]:
                cex_peak_count_list.append(len(shower_dir))

        selected_mask = ak.Array(shower_selection_mask)

        print("nSP Hist:", np.histogram(cex_nsp_list, range=[0, 200], bins=20))
        print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0", selected_mask], return_counts=True))

        print("Shower Count: 1/2/3/4/5 =",
              np.sum(np.asarray(peak_count_list) == 1), "/",
              np.sum(np.asarray(peak_count_list) == 2), "/",
              np.sum(np.asarray(peak_count_list) == 3), "/",
              np.sum(np.asarray(peak_count_list) == 4), "/",
              np.sum(np.asarray(peak_count_list) == 5))

        print("sCEX Shower Count: 0/1/2/3/4/5 =",
              np.sum(np.asarray(cex_peak_count_list) == 0), "/",
              np.sum(np.asarray(cex_peak_count_list) == 1), "/",
              np.sum(np.asarray(cex_peak_count_list) == 2), "/",
              np.sum(np.asarray(cex_peak_count_list) == 3), "/",
              np.sum(np.asarray(cex_peak_count_list) == 4), "/",
              np.sum(np.asarray(cex_peak_count_list) == 5))

        print("sCEX PI0 Count LL: 0/1/2 =",
              np.sum(np.asarray(cex_npi0_list) == 0), "/",
              np.sum(np.asarray(cex_npi0_list) == 1), "/",
              np.sum(np.asarray(cex_npi0_list) == 2))

        events["two_shower_dir"] = ak.Array(peak_angles)

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
            hists.plot_process_stack(x=events, idx=idx, variable=plot, precut=precut)
            hists.plot_particles_stack(x=events[plot], x_pdg=pdg, idx=idx, precut=precut)
            hists.plot_particles(x=events[plot], idx=idx, precut=precut)

    def efficiency(self, total_events, passed_events, cut, hists):
        for idx, plot in enumerate(self.local_hist_config):
            hists.plot_efficiency(xtotal=total_events[plot], xpassed=passed_events[plot], idx=idx)

    def cut_optimization(self):
        # get cut values and iterate to the next
        values = self.optimization_values.__next__()
        print("VALUES", values)
        self.ll.likelihood_l10_cut = values[0]
        self.ll.likelihood_l12_cut = values[1]

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
