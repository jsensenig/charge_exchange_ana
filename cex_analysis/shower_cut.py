from cex_analysis.event_selection_base import EventSelectionBase
import awkward as ak
import numpy as np
import tmp.shower_count_direction as sdir
import tmp.shower_likelihood as sll


class ShowerCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "ShowerCut"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

        # FIXME test shower counting, make local class object
        self.dir = sdir.ShowerDirection()
        self.ll = sll.ShowerLikelihood()

    def cnn_shower_cut(self, events):
        # Create a mask for all daughters with CNN EM-like score <0.5
        return events[self.local_config["track_like_cnn_var"]] < self.local_config["cnn_shower_cut"]

    def min_shower_energy_cut(self, events):
        return events[self.local_config["shower_energy_var"]] > self.local_config["small_energy_shower_cut"]

    def shower_count_cut(self, events):
        """
        1. CNN cut to select shower-like daughters
        2. Energy cut, eliminate small showers from e.g. de-excitation gammas
        :param events:
        :return:
        """
        # Perform a 2 step cut on showers, get a daughter mask from each
        cnn_shower_mask = self.cnn_shower_cut(events)
        min_shower_energy = self.min_shower_energy_cut(events)
        nhit_mask = events["reco_daughter_PFP_nHits"] > 80.

        # Shower selection mask
        shower_mask = cnn_shower_mask & min_shower_energy & nhit_mask

        # We want to count the number of potential showers in each event
        shower_count = np.count_nonzero(events[self.local_config["shower_energy_var"], shower_mask], axis=1)

        shower_print_cex = shower_count[events["single_charge_exchange"]]
        print("sCEX Old Shower Count: 0/1/2/3/4/5 =",
              ak.count_nonzero(shower_print_cex == 0), "/",
              ak.sum(shower_print_cex == 1), "/",
              ak.sum(shower_print_cex == 2), "/",
              ak.sum(shower_print_cex == 3), "/",
              ak.sum(shower_print_cex == 4), "/",
              ak.sum(shower_print_cex == 5))

        shower_print_pi0_prod = shower_count[events["pi0_production"]]
        print("Pi0 Prod Old Shower Count: 0/1/2/3/4/5 =",
              ak.count_nonzero(shower_print_pi0_prod == 0), "/",
              ak.sum(shower_print_pi0_prod == 1), "/",
              ak.sum(shower_print_pi0_prod == 2), "/",
              ak.sum(shower_print_pi0_prod == 3), "/",
              ak.sum(shower_print_pi0_prod == 4), "/",
              ak.sum(shower_print_pi0_prod == 5))

        # Create the event mask, true if there are 2 candidate showers
        # return shower_count > 0
        #return shower_count < 3
        return (shower_count == 2) | (shower_count == 1)

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # The variable on which we cut
        cut_variable = self.local_config["cut_variable"]

        # Plot the variable before making cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Candidate shower count mask
        selected_mask = self.shower_count_cut(events)

        ###############################
        # Method which counts the peaks in theta-phi histogram as a proxy for
        # number of showers.
        use_new_method = True
        if use_new_method:
            peak_count_list = []
            cex_peak_count_list = []
            cex_npi0_list = []
            cex_nsp_list = []
            peak_angles = []
            shower_selection_mask = []
            for i in range(0, len(events)):
                if events[i] is None or len(events["reco_daughter_PFP_shower_spacePts_X", i]) < 50:#events["reco_daughter_PFP_shower_spacePts_count", i] < 10:
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
                pred_npi0 = self.ll.classify_npi0(coord[:, 3])
                #shower_selection_mask.append(True)
                if pred_npi0 == 1 or (pred_npi0 != 1 and peak_count == 2) or (pred_npi0 == 2 and peak_count == 1):
                #if pred_npi0 == 1 or (pred_npi0 != 1 and peak_count == 2) or (pred_npi0 == 2 and peak_count == 1) or (pred_npi0 == 0 and peak_count > 2):
                    shower_selection_mask.append(True)
                    if events["single_charge_exchange", i]:
                        cex_npi0_list.append(1)
                        cex_nsp_list.append(events["reco_daughter_PFP_shower_spacePts_count", i])
                else:
                    shower_selection_mask.append(False)
                    if events["single_charge_exchange", i]:
                        cex_npi0_list.append(pred_npi0)
                ##############
                peak_count_list.append(len(shower_dir))
                if events["single_charge_exchange", i]:
                    cex_peak_count_list.append(len(shower_dir))

            # selected_mask = (np.asarray(peak_count_list) == 1) | (np.asarray(peak_count_list) == 2)
            # selected_mask = (np.asarray(peak_count_list) == 2)

            # Unfortunately the number of Pandora showers introduces None's into the
            # array which can't be handled downstream. To remedy this we replace all
            # None's with 'True' (this is a bool array). We only want to use the the
            # Pandora selection to help remove impurities while avoiding it's inefficiency
            # therefore we replace None's (unknown) with True.
            selected_mask = np.where(ak.is_none(selected_mask), True, selected_mask)

            # 'AND' the Pandora selection with the Likelihood selection array
            #selected_mask = ak.Array(shower_selection_mask) & selected_mask
            selected_mask = ak.Array(shower_selection_mask) & selected_mask

            print("nSP Hist:", np.histogram(cex_nsp_list, range=[0, 200], bins=20))
            print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0"], return_counts=True))

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
        ###############################

        # selected_mask = selected_mask & (events["reco_daughter_PFP_nHits"] > 50.)

        # TODO Add pi0 kinematics calculation here

        # Plot the variable after cut
        self.plot_particles_base(events=events[selected_mask], pdg=events[self.reco_beam_pdg, selected_mask],
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

    def get_cut_doc(self):
        doc_string = "Cut on daughter showers"
        return doc_string
