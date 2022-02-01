from cex_analysis.event_handler import EventHandler
from cex_analysis.true_process import TrueProcess
import awkward as ak
from uproot import concatenate
import json

from bayes_opt import BayesianOptimization


class OptimizationHandler:
    def __init__(self, config):

        self.config = config

        # Now assemble our cuts
        self.event_handler = EventHandler(config)
        self.cut_map = self.event_handler.cut_map

        self.beam_cut_map = self.config["beam_cut_list"]
        self.selection_param_map = {}

        self._events = None
        self._total_true_scex_events = 0

    def get_analysis_parameters(self):
        for cut in self.cut_map:
            if cut in self.beam_cut_map:
                continue
            # Now loop through each cut's parameters to find the ones ending in 'param'
            for key in self.cut_map[cut].local_config:
                if key[-5:] == 'param':
                    print("PARAMETER: ", str(cut) + "_" + key, " ", self.cut_map[cut].local_config[key])
                    self.selection_param_map[str(cut) + "_" + key] = self.cut_map[cut].local_config[key]

    def run_beam_preselection(self, events):
        """
        Here we run the beam preselection to select primary pi+.
        :param events: Array of events to be analyzed
        :return:
        """
        for i, cut in enumerate(self.cut_map):
            if cut in self.beam_cut_map:
                # Mask out events which are not selected
                events = events[event_mask] if i > 0 else events
                # Perform the cut selection, we don't need the histograms in optimization mode
                event_mask = self.cut_map[cut].selection(events, None, optimizing=True)

        return events

    def configure_cuts(self, param_map):
        for cut in self.cut_map:
            # Now loop through each cut's parameters to find the ones ending in 'param'
            for key in self.cut_map[cut].local_config:
                if key[-5:] == 'param':
                    self.cut_map[cut].local_config[key] = param_map[str(cut) + "_" + key]

    def run_selection_optimization(self, **kwargs):
        """
        This will be our f(x) to be optimized.
        Here we run the selection. This means we loop over each cut
        passing the data and receving a selection mask from it in
        sequential order.
        :param events: Array of events to be analyzed
        :return:
        """
        # Reconfigure cut values
        self.configure_cuts(kwargs)

        for i, cut in enumerate(self.cut_map):
            print("CUT:", cut)
            # Mask out events not selected
            events = events[event_mask] if i > 0 else self._events
            # Perform the cut selection, we don't need the histograms in optimization mode
            event_mask = self.cut_map[cut].selection(events, None, optimizing=True)

        # Find the signal and background counts
        num_true_selected = ak.count_nonzero(events["single_charge_exchange", event_mask], axis=0)
        num_total_selected = ak.count_nonzero(event_mask, axis=0)

        print("TRUE/TOTAL", num_true_selected, "/", num_total_selected)
        print("Eff:", (num_true_selected/self._total_true_scex_events), "Purity:", (num_true_selected/num_total_selected))

        # Return S/sqrt(B)
        #return num_true_selected / ak.numpy.sqrt(num_total_selected - num_true_selected)
        # Return S/B
        #return num_true_selected / (num_total_selected - num_true_selected)
        # Return S/sqrt(S+B)
        #return num_true_selected / ak.numpy.sqrt(num_total_selected)
        # Return e*p
        #return (num_true_selected / self._total_true_scex_events) * (num_true_selected / num_total_selected)
        # Full FOM from likelihood
        bkgd = num_total_selected - num_true_selected
        return ak.numpy.sqrt(2.*(num_true_selected + bkgd)*ak.numpy.log(1+(num_true_selected/bkgd))-2.*num_true_selected)

    def load_data(self):

        branches = ["event", "reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG",
                    "reco_beam_passes_beam_cuts", "reco_daughter_PFP_true_byHits_parPDG", "true_beam_daughter_startP",
                    "true_beam_daughter_PDG", "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy",
                    "reco_daughter_PFP_trackScore_collection", "reco_daughter_allTrack_Chi2_proton",
                    "reco_daughter_allTrack_Chi2_ndof", "beam_inst_TOF", "true_daughter_nPiMinus",
                    "true_daughter_nPiPlus", "true_daughter_nPi0", "true_daughter_nProton", "true_daughter_nNeutron",
                    "true_beam_PDG", "true_beam_endProcess", "true_beam_endZ", "true_beam_endP", "true_beam_endPx",
                    "true_beam_endPy", "true_beam_endPz", "reco_daughter_PFP_michelScore_collection",
                    "reco_beam_calo_startX", "reco_beam_calo_startY", "reco_beam_calo_startZ", "reco_beam_calo_endX",
                    "reco_beam_calo_endY", "reco_beam_calo_endZ", "true_beam_daughter_startPx",
                    "true_beam_daughter_startPy", "true_beam_daughter_startPz", "reco_beam_true_byHits_endProcess",
                    "reco_daughter_PFP_nHits", "beam_inst_P", "reco_beam_vertex_michel_score", "reco_beam_vertex_nHits",
                    "reco_daughter_allTrack_dEdX_SCE", "reco_beam_endX", "reco_beam_endY", "reco_beam_endZ"]

        # Space-point branches
        branches += ["reco_daughter_PFP_shower_spacePts_X", "reco_daughter_PFP_shower_spacePts_Y",
                     "reco_daughter_PFP_shower_spacePts_Z", "reco_daughter_PFP_shower_spacePts_count",
                     "reco_daughter_PFP_shower_spacePts_E"]
                     #"reco_daughter_PFP_shower_spacePts_gmother_ID", "reco_daughter_PFP_shower_spacePts_mother_ID",
                     #"reco_daughter_PFP_shower_spacePts_gmother_PDG"]

        branches += ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz",
                     "true_beam_Pi0_decay_PDG", "true_beam_Pi0_decay_ID", "true_beam_Pi0_decay_startP",
                     "true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ",
                     "true_beam_endX", "true_beam_endY", "true_beam_endZ", "reco_beam_trackEndDirX",
                     "reco_beam_trackEndDirY", "reco_beam_trackEndDirZ", "true_beam_interactingEnergy",
                     "true_beam_incidentEnergies", "dEdX_truncated_mean"]

        branches += ["reco_beam_calo_startDirX", "reco_beam_calo_startDirY", "reco_beam_calo_startDirZ",
                     "reco_beam_calo_endDirX", "reco_beam_calo_endDirY", "reco_beam_calo_endDirZ"]

        file_name = "/Users/jsen/tmp/pion_qe/cex_selection/macros/merge_files/full_mc_merged_n86k_new_inc_tmean_branch.root:beamana;121"

        return concatenate(file_name, branches)

    def run_optimization(self):
        print("Starting Optimization")

        # Load data from file
        events = self.load_data()
        #events = events[0:5000]

        # Run beam preselection to select only beam pi+ events
        self._events = self.run_beam_preselection(events)

        # Add the interaction process columns
        classify_process = TrueProcess()
        self._events = classify_process.classify_event_process(events=self._events)
        # Get total number of true signal events
        self._total_true_scex_events = ak.count_nonzero(self._events["single_charge_exchange"], axis=0)

        # Get parameters from cuts to be optimized
        self.get_analysis_parameters()

        #print("S/sqrt(B) = ", self.run_selection_optimization(events))

        # Bounded region of parameter space
        pbounds = {'ShowerPreCut_cnn_shower_cut_param': (0.2, 0.6), 'ShowerPreCut_small_energy_shower_cut_param': (20, 90),
         'ShowerCut_likelihood_l10_param': (0.5, 1.5), 'ShowerCut_likelihood_l12_param': (0.5, 2.0),
         'DaughterPionCut_chi2_ndof_cut_param': (50, 150), 'DaughterPionCut_cnn_track_cut_param': (0.2, 0.7),
         'MichelDaughterCut_cnn_michel_cut_param': (0.6, 0.95), 'TruncatedDedxCut_cnn_track_cut_param': (0.2, 0.6),
         'TruncatedDedxCut_lower_trunc_mean_param': (0.5, 1.5), 'TruncatedDedxCut_upper_trunc_mean_param': (2.0, 3.0)}

        optimizer = BayesianOptimization(f=self.run_selection_optimization,
                                         pbounds=pbounds,
                                         random_state=1)

        optimizer.maximize(init_points=4,
                           n_iter=5)

        print("MAX RESULTS: ", optimizer.max)


if __name__ == "__main__":

    with open("config/bayesian_optimizer.json", "r") as cfg:
        config = json.load(cfg)

    opt = OptimizationHandler(config)
    opt.run_optimization()
