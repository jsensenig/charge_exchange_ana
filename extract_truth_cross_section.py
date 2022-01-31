from cross_section.cex_dd_cross_section import CexDDCrossSection
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
            "reco_beam_vertex_nHits", "true_beam_interactingEnergy", "true_beam_incidentEnergies", "true_beam_slices",
            "true_beam_traj_incidentEnergies", "true_beam_traj_interacting_Energy"]

# Single beam pi+ events
#files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana*.root"
#all_events = uproot.concatenate(files={files:"pduneana/beamana;1"}, expressions=branches)

#files = "/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_unique_sample_n100k/full_mc_sp_merged_unique_n86k.root"
files = "/Users/jsen/tmp/pion_qe/cex_selection/macros/merge_files/full_mc_merged_n86k_new_inc_int_branch.root"
all_events = uproot.concatenate(files={files:"beamana;121"}, expressions=branches)

xsec = CexDDCrossSection(None)
xsec.extract_cross_section(all_events=all_events, selected_events=all_events, total_incident_pion=14000)

