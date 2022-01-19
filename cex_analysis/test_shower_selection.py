from cex_analysis.event_selection_base import EventSelectionBase
from cex_analysis.true_process import TrueProcess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tmp.shower_count_direction as sdir
import tmp.shower_likelihood as sll

import awkward as ak
import numpy as np

class TestShowerSelection(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

        self.cut_name = "TestShowerSelection"
        self.config = config
        self.reco_beam_pdg = self.config["reco_daughter_pdg"]

        # Configure class
        self.local_config, self.local_hist_config = super().configure(config_file=self.config[self.cut_name]["config_file"],
                                                                      cut_name=self.cut_name)

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

        scex_selected_count = np.count_nonzero(events["single_charge_exchange"][shower_count == 2])
        scex_total_count = np.count_nonzero(events["single_charge_exchange"])
        selected_total_count = np.count_nonzero(shower_count == 2)

        print("sCEX Old Shower Efficiency", scex_selected_count / scex_total_count)
        print("sCEX Old Shower Purity", scex_selected_count / selected_total_count)

        #true_total_pi0_count = np.count_nonzero(events["true_beam_grand_daughter_PDG"] == 111, axis=1) + events["true_daughter_nPi0"]
        true_total_pi0_count = events["true_daughter_nPi0"]

        bin_cnt, _, _, _ = plt.hist2d(shower_count, true_total_pi0_count, range=[[0, 10],[0, 5]], bins=[10, 5], cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot([0, 10], [0, 5])
        plt.xlabel('Pandora nShowers')
        plt.ylabel('True $n\pi^0$')
        plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/true_reco_pandora_nshowers.png")
        plt.close()
        # print("Pi0 COS THETA", np.sum(cnt), "/", processed_event)

        bin_cnt = bin_cnt.T
        print("PANDORA HIST", bin_cnt)

        print("0 Pi0 Pred. Efficiency", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[0, :]), " [%]")
        print("0 Pi0 Pred. Purity", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[:, 0]), " [%]")

        print("1 Pi0 Pred. Efficiency", 100. * bin_cnt[1, 2] / np.sum(bin_cnt[1, :]), " [%]")
        print("1 Pi0 Pred. Purity", 100. * bin_cnt[1, 2] / np.sum(bin_cnt[:, 2]), " [%]")

        print("2 Pi0 Pred. Efficiency", 100. * bin_cnt[2, 4] / np.sum(bin_cnt[2, :]), " [%]")
        print("2 Pi0 Pred. Purity", 100. * bin_cnt[2, 4] / np.sum(bin_cnt[:, 4]), " [%]")

        # Create the event mask, true if there are 2 candidate showers
        # return shower_count > 0
        #return shower_count < 3
        return shower_count == 2

    def selection(self, events, hists):
        # First we configure the histograms we want to make
        hists.configure_hists(self.local_hist_config)

        # Plot the variable before making cut
        self.plot_particles_base(events=events, pdg=events[self.reco_beam_pdg], precut=True, hists=hists)

        # Give the events an ordered and unique index
        events["event_index"] = np.arange(0, len(events))

        # Candidate shower count mask
        selected_mask = self.shower_count_cut(events)

        ###############################
        # Method which counts the peaks in theta-phi histogram as a proxy for
        # number of showers.
        use_new_method = True
        if use_new_method:
            peak_count_list = []
            all_npi0_list = []
            predicted_npi0_list = []
            likelihood_list = []
            selected_npi0_list = []
            cex_peak_count_list = []
            cex_npi0_list = []
            cex_nsp_list = []
            peak_angles = []
            shower_selection_mask = []
            ll_shower_selection_mask = []
            hist_shower_selection_mask = []
            for i in range(0, len(events)):
                pi0_counts = np.count_nonzero(events["true_beam_grand_daughter_PDG", i] == 111) if events[i] is not None else 0
                #all_npi0_list.append(0) if events[i] is None else all_npi0_list.append(events["true_daughter_nPi0", i]+pi0_counts)
                all_npi0_list.append(0) if events[i] is None else all_npi0_list.append(events["true_daughter_nPi0", i])
                if events[i] is None or len(events["reco_daughter_PFP_shower_spacePts_X", i]) < 50:#events["reco_daughter_PFP_shower_spacePts_count", i] < 10:
                    if events["single_charge_exchange", i]:
                        cex_peak_count_list.append(0)
                        cex_npi0_list.append(0)
                    predicted_npi0_list.append(0)
                    selected_npi0_list.append(0)
                    shower_selection_mask.append(False)
                    ll_shower_selection_mask.append(False)
                    hist_shower_selection_mask.append(False)
                    peak_count_list.append(0)
                    peak_angles.append(np.array([]))
                    likelihood_list.append(np.array([-10., -10., -10.]))
                    continue
                coord = self.dir.transform_to_spherical(events=events[i])
                if coord is None:
                    if events["single_charge_exchange", i]:
                        cex_peak_count_list.append(0)
                        cex_npi0_list.append(0)
                    predicted_npi0_list.append(0)
                    selected_npi0_list.append(0)
                    shower_selection_mask.append(False)
                    ll_shower_selection_mask.append(False)
                    hist_shower_selection_mask.append(False)
                    peak_count_list.append(0)
                    peak_angles.append(np.array([]))
                    likelihood_list.append(np.array([-10.,-10.,-10.]))
                    continue
                # For the peaks cut within 3.5 radiation lengths
                rmask = coord[:, 3] <= (3.5 * 14.)  # 3.5 * X_0
                shower_dir = []
                if np.count_nonzero(rmask) > 0:
                    shower_dir = self.dir.get_shower_direction_unit(coord[rmask])
                    # shower_dir = self.dir.get_shower_direction_unit(coord)
                peak_angles.append(shower_dir)
                peak_count = len(shower_dir)
                pred_npi0, ll = self.ll.classify_npi0(coord[:, 3], return_likelihood=True)
                likelihood_list.append(ll)
                if peak_count == 2:
                    hist_shower_selection_mask.append(True)
                else:
                    hist_shower_selection_mask.append(False)
                if pred_npi0 == 1:
                    ll_shower_selection_mask.append(True)
                else:
                    ll_shower_selection_mask.append(False)
                if pred_npi0 == 1 or (peak_count == 2) or (pred_npi0 == 2 and peak_count == 1) or (pred_npi0 == 0 and peak_count > 2):
                #if (pred_npi0 == 1) or (pred_npi0 == 2 and peak_count == 1):
                #if pred_npi0 == 1 or (pred_npi0 != 1 and peak_count == 2):
                    shower_selection_mask.append(True)
                    selected_npi0_list.append(1)
                    if events["single_charge_exchange", i]:
                        cex_npi0_list.append(1)
                        cex_nsp_list.append(events["reco_daughter_PFP_shower_spacePts_count", i])
                else:
                    shower_selection_mask.append(False)
                    selected_npi0_list.append(0)
                    if events["single_charge_exchange", i]:
                        cex_npi0_list.append(pred_npi0)
                ##############
                peak_count_list.append(len(shower_dir))
                predicted_npi0_list.append(pred_npi0)

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
            selected_mask = ak.Array(shower_selection_mask) #| selected_mask

            print("CHECK EFF", np.count_nonzero(selected_mask & (events["true_daughter_nPi0"] == 1))/np.count_nonzero((events["true_daughter_nPi0"] == 1)))
            print("CHECK Purity", np.count_nonzero(selected_mask & (events["true_daughter_nPi0"] == 1))/np.count_nonzero(selected_mask))

            pandora_selected_mask = selected_mask & (events["true_daughter_nPi0"] == 1)
            #print("PANDORA MASK LEN / IDX LEN", len(selected_mask), "/", len(pandora_selected_idx))

            hist_ll_selected_mask = ak.Array(shower_selection_mask) & (events["true_daughter_nPi0"] == 1) #& selected_mask
            hist_selected_mask = ak.Array(hist_shower_selection_mask) & (events["true_daughter_nPi0"] == 1)
            ll_selected_mask = ak.Array(ll_shower_selection_mask) & (events["true_daughter_nPi0"] == 1)

            print("Pi0 > 0 Count", np.count_nonzero((events["true_daughter_nPi0"] > 0)))

            new_method_selected_mask = hist_selected_mask & ll_selected_mask & (events["true_daughter_nPi0"] == 1)
            print("Hist-LL MASK LEN", len(new_method_selected_mask))
            print("Hist-LL Non-Zero Count", np.count_nonzero(new_method_selected_mask))
            print("Pandora Non-Zero Count", np.count_nonzero(pandora_selected_mask))
            print("Hist Non-Zero Count", np.count_nonzero(hist_selected_mask))
            print("LL Non-Zero Count", np.count_nonzero(ll_selected_mask))
            print("Hist+LL Non-Zero Count", np.count_nonzero(hist_ll_selected_mask))

            new_method_selected_mask = hist_selected_mask & pandora_selected_mask & (events["true_daughter_nPi0"] == 1)
            print("Hist-Pandora MASK LEN", len(new_method_selected_mask))
            print("Hist-Pandora Non-Zero Count", np.count_nonzero(new_method_selected_mask))

            new_method_selected_mask = hist_ll_selected_mask & pandora_selected_mask & (events["true_daughter_nPi0"] == 1)
            print("Hist+LL-Pandora MASK LEN", len(new_method_selected_mask))
            print("Hist+LL-Pandora Non-Zero Count", np.count_nonzero(new_method_selected_mask))

            new_method_selected_mask = ll_selected_mask & pandora_selected_mask & (events["true_daughter_nPi0"] == 1)
            print("LL-Pandora MASK LEN", len(new_method_selected_mask))
            print("LL-Pandora Non-Zero Count", np.count_nonzero(new_method_selected_mask))

            # Get the overlap between selections
            selection_overlap = new_method_selected_mask & pandora_selected_mask

            print("nSP Hist:", np.histogram(cex_nsp_list, range=[0, 200], bins=20))
            print("Selected Events True nPi0: ", np.unique(events["true_daughter_nPi0"], return_counts=True))
            #pi0_grand_daughter_mask = events["true_beam_grand_daughter_PDG"] == 111
            pi0_counts = np.count_nonzero(events["true_beam_grand_daughter_PDG"] == 111, axis=1)
            print("PI0 COUNT TYPE", type(pi0_counts))
            print("PI0 COUNT SHAPE", pi0_counts.type, " - ", events["true_daughter_nPi0"].type)
            print("Selected Events True GD nPi0: ", np.unique(pi0_counts, return_counts=True))

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

        # TODO MAKE PLOTSSS
        """
        1. 2d Histogram of true vs selected nPi0
            i) For LL
            ii) Pandora
        2. Pandora nShowers vs nPi0
        """
        if use_new_method:
            bin_cnt, _, _, _ = plt.hist2d(predicted_npi0_list, all_npi0_list, range=[[0, 3],[0, 5]], bins=[3, 5], cmap=plt.cm.jet)
            plt.colorbar()
            plt.plot([0, 3], [0, 3])
            plt.xlabel('LL Pred. $n\pi^0$')
            plt.ylabel('True $n\pi^0$')
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/true_reco_predicted_npi0.png")
            plt.close()
            # print("Pi0 COS THETA", np.sum(cnt), "/", processed_event)

            # Transpose so we have shape Row,Col
            bin_cnt = bin_cnt.T

            print("LL HIST SHAPE", bin_cnt.shape)
            print("LL HIST", bin_cnt)

            print("0 Pi0 Pred. Efficiency #", bin_cnt[0, 0], "/", np.sum(bin_cnt[0, :]))
            print("0 Pi0 Pred. Purity #", bin_cnt[0, 0], "/", np.sum(bin_cnt[:, 0]))
            print("0 Pi0 Pred. Efficiency", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[0, :]), " [%]")
            print("0 Pi0 Pred. Purity", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[:, 0]), " [%]")

            print("1 Pi0 Pred. Efficiency #", bin_cnt[1, 1], "/", np.sum(bin_cnt[1, :]))
            print("1 Pi0 Pred. Purity #", bin_cnt[1, 1], "/", np.sum(bin_cnt[:, 1]))
            print("1 Pi0 Pred. Efficiency", 100. * bin_cnt[1, 1] / np.sum(bin_cnt[1, :]), " [%]")
            print("1 Pi0 Pred. Purity", 100. * bin_cnt[1, 1] / np.sum(bin_cnt[:, 1]), " [%]")

            print(">1 Pi0 Pred. Efficiency #", bin_cnt[2:-1, 2], "/", np.sum(bin_cnt[2:-1, :]))
            print(">1 Pi0 Pred. Purity #", bin_cnt[2:-1, 2], "/", np.sum(bin_cnt[:, 2]))
            print(">1 Pi0 Pred. Efficiency", 100. * bin_cnt[2:-1, 2] / np.sum(bin_cnt[2:-1, :]), " [%]")
            print(">1 Pi0 Pred. Purity", 100. * bin_cnt[2:-1, 2] / np.sum(bin_cnt[:, 2]), " [%]")

            bin_cnt, _, _, _ = plt.hist2d(selected_npi0_list, all_npi0_list, range=[[0, 2],[0, 5]], bins=[2, 5], cmap=plt.cm.jet)
            plt.colorbar()
            plt.xlabel('LL + Peak Selected $n\pi^0$')
            plt.ylabel('True $n\pi^0$')
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/true_reco_selected_npi0.png")
            plt.close()

            bin_cnt = bin_cnt.T

            print("0 Pi0 Selected Efficiency", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[0, :]), " [%]")
            print("0 Pi0 Selected Purity", 100. * bin_cnt[0, 0] / np.sum(bin_cnt[:, 0]), " [%]")

            print("1 Pi0 Selected Efficiency", 100. * bin_cnt[1, 1] / np.sum(bin_cnt[1, :]), " [%]")
            print("1 Pi0 Selected Purity", 100. * bin_cnt[1, 1] / np.sum(bin_cnt[:, 1]), " [%]")

            bin_cnt, _, _ = plt.hist(selection_overlap, range=[0, 2], bins=2)
            plt.xlabel('New Method - Pandora Selection Overlap')
            plt.ylabel('Count')
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/selection_overlap.png")
            plt.close()
            print("OVERLAP BINS", bin_cnt)

            all_npi0_list = np.asarray(all_npi0_list)
            likelihood_list = np.asarray(likelihood_list)

            print("PRE-HIST SHAPE", all_npi0_list.shape, "/", likelihood_list[:, 1].shape)

            bin_cnt, _, _, _ = plt.hist2d(likelihood_list[:, 2], all_npi0_list, range=[[2,102],[0,4]], bins=[50, 4], cmap=plt.cm.jet)
            plt.colorbar()
            plt.xlabel('-2ln$\lambda$')
            plt.ylabel('True $n\pi^0$')
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/likelihood_vs_true_npi0.png")
            plt.close()

            bin_cnt0, _, _ = plt.hist(likelihood_list[:, 0][all_npi0_list == 0], facecolor='None', edgecolor='blue', range=[2,132], bins=65)
            bin_cnt, _, _ = plt.hist(likelihood_list[:, 1][all_npi0_list == 0], facecolor='None', edgecolor='red', range=[2, 132], bins=65)
            bin_cnt, _, _ = plt.hist(likelihood_list[:, 2][all_npi0_list == 0], facecolor='None', edgecolor='green', range=[2, 132], bins=65)
            plt.xlabel('-2ln$\lambda$', fontsize=15)
            plt.ylabel('Count', fontsize=15)
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/likelihood_0pi0.png")
            plt.close()
            print("BIN COUNT0", bin_cnt0)

            bin_cnt, _, _ = plt.hist(likelihood_list[:, 0][all_npi0_list == 1], facecolor='None', edgecolor='blue', range=[2,102], bins=50)
            bin_cnt1, _, _ = plt.hist(likelihood_list[:, 1][all_npi0_list == 1], facecolor='None', edgecolor='red', range=[2, 102], bins=50)
            bin_cnt, _, _ = plt.hist(likelihood_list[:, 2][all_npi0_list == 1], facecolor='None', edgecolor='green', range=[2, 102], bins=50)
            plt.xlabel('-2ln$\lambda$', fontsize=15)
            plt.ylabel('Count', fontsize=15)
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/likelihood_1pi0.png")
            plt.close()
            print("BIN COUNT1", bin_cnt1)

            bin_cnt, _, _ = plt.hist(likelihood_list[:, 0][all_npi0_list > 1], facecolor='None', edgecolor='blue', range=[2,102], bins=50)
            bin_cnt, _, _ = plt.hist(likelihood_list[:, 1][all_npi0_list > 1], facecolor='None', edgecolor='red', range=[2, 102], bins=50)
            bin_cnt2, _, _ = plt.hist(likelihood_list[:, 2][all_npi0_list > 1], facecolor='None', edgecolor='green', range=[2, 102], bins=50)
            plt.xlabel('-2ln$\lambda$', fontsize=15)
            plt.ylabel('Count', fontsize=15)
            plt.savefig("/Users/jsen/work/Protodune/analysis/test_shower_selection/charge_exchange_ana/figs/likelihood_npi0.png")
            plt.close()
            print("BIN COUNT2", bin_cnt2)

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

