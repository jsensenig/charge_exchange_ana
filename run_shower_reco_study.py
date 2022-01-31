from macros.shower_direction_study.reco_shower_direction import RecoShowerDirection
import uproot


branches = ["event", "reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG", "reco_beam_passes_beam_cuts",
            "reco_daughter_PFP_true_byHits_parPDG", "true_beam_daughter_startP", "true_beam_daughter_PDG",
            "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy", "reco_daughter_PFP_trackScore_collection",
            "reco_daughter_allTrack_Chi2_proton", "reco_daughter_allTrack_Chi2_ndof",
            "true_daughter_nPiMinus", "true_daughter_nPiPlus", "true_daughter_nPi0", "true_daughter_nProton",
            "true_daughter_nNeutron","true_beam_PDG","true_beam_endProcess","true_beam_endZ","true_beam_endP","true_beam_endPx",
            "true_beam_endPy", "true_beam_endPz", "reco_daughter_PFP_michelScore_collection", "reco_beam_calo_startX",
            "reco_beam_calo_startY", "reco_beam_calo_startZ", "reco_beam_calo_endX", "reco_beam_calo_endY",
            "reco_beam_calo_endZ", "true_beam_daughter_startPx", "true_beam_daughter_startPy", "true_beam_daughter_startPz",
            "reco_beam_true_byHits_endProcess", "reco_daughter_PFP_nHits", "reco_beam_vertex_michel_score",
            "reco_beam_vertex_nHits"]

branches += ["reco_daughter_PFP_shower_spacePts_X", "reco_daughter_PFP_shower_spacePts_Y",
             "reco_daughter_PFP_shower_spacePts_Z", "reco_daughter_PFP_shower_spacePts_count",
             "reco_daughter_PFP_shower_spacePts_gmother_ID", "reco_daughter_PFP_shower_spacePts_mother_ID",
             "reco_daughter_PFP_shower_spacePts_gmother_PDG"]

branches += ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz",
             "true_beam_Pi0_decay_PDG", "true_beam_Pi0_decay_ID", "true_beam_Pi0_decay_startP"]

# Single beam pi+ events
files = "/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_unique_sample_n14000/full_mc_shower_sp_merged_unique_n13500.root"

all_events = uproot.concatenate(files={files:"beamana;19"}, expressions=branches)
shower_reco = RecoShowerDirection()
shower_reco.run_shower_reco(events=all_events)
