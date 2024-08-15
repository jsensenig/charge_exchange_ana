from timeit import default_timer as timer
from cex_analysis.event_handler import EventHandler
import cex_analysis.efficiency_data as eff_data
from unfolding.unfolding_interface import BeamPionVariables, Pi0Variables
from unfolding.unfold_events import Unfold


import numpy as np
import uproot
import time
import json
import h5py
import os


def open_file(file_name, tree_dir):
    try:
        file1 = uproot.open(file_name)
        tree = file1[tree_dir]
    except FileNotFoundError:
        print("Could not find file", file_name)
        return
    return tree


def calculate_efficiency(selected_true, selected_total, true_count_list):
    #total_eff, total_sel, total_count = eff_data.combine_efficiency(cut_efficiency_dict_list=selected_true,
    #                                                                cut_total_dict_list=selected_total,
    #                                                                process_true_count_list=true_count_list)

    cum_eff, purity, eff, sel, fom = eff_data.calculate_efficiency(cut_efficiency_dict=selected_true,
                                                                   cut_total_dict=selected_total,
                                                                   process_true_count=true_count_list)

    for ceff, f, p, cut, s, l in zip(cum_eff, eff, purity, selected_true, sel, fom):
        print("Cut: [\033[92m", '{:<18}'.format(cut), "\033[0m] Cumulative eff:", '{:.4f}'.format(ceff),
              " Eff:", '{:.4f}'.format(f), "Purity:", '{:.4f}'.format(p),
              "S/sqrt(S+B):", '{:.4f}'.format(l), "Selection:", s, "True/Total")


def save_results(results, is_mc):

    hist_map, event_mask, cut_signal_selected, cut_total_selected, signal_total, events, beam_events = results

    cut_names = []
    cut_total = np.count_nonzero(event_mask)
    cut_signal = 0

    if is_mc:
        calculate_efficiency(cut_signal_selected, cut_total_selected, signal_total)

        *_, num_last_cut = cut_signal_selected.items()
        print("Selected", num_last_cut, " events out of", signal_total, "true CEX events")

        cut_names = list(cut_total_selected.keys())
        cut_total = list(cut_total_selected.values())
        cut_signal = list(cut_signal_selected.values())

    print("Total events selected:", cut_total)

    hist_file = 'test_hist_file.hdf5'
    cut_file = 'test_cut_file.hdf5'
    os.remove(hist_file) if os.path.exists(hist_file) else None
    os.remove(cut_file) if os.path.exists(cut_file) else None


    # Write cut efficiency and purity
    h5_file = h5py.File('test_cut_file.hdf5', 'w')
    h5_file.create_dataset('cut_names', data=cut_names)
    h5_file.create_dataset('total_signal', data=signal_total)
    h5_file.create_dataset('cut_total', data=cut_total)
    h5_file.create_dataset('cut_signal', data=cut_signal)
    h5_file.create_dataset('selection_mask', data=event_mask)
    h5_file.close()

    num_hists = len(hist_map)

    # Open file and declare data types
    h5_file = h5py.File('test_hist_file.hdf5', 'w')
    data_str = h5py.string_dtype(encoding='utf-8')
    data_vlen_str = h5py.vlen_dtype(data_str)
    data_float32 = h5py.vlen_dtype(np.dtype('float32'))
    data_int32 = h5py.vlen_dtype(np.dtype('int32'))

    # Top level, one for each histogram
    hist_name = h5_file.create_dataset('hist_name', (num_hists,), dtype=data_str)
    hist_type = h5_file.create_dataset('hist_type', (num_hists,), dtype=data_str)
    hist_xlabel = h5_file.create_dataset('hist_xlabel', (num_hists,), dtype=data_str)
    hist_ylabel = h5_file.create_dataset('hist_ylabel', (num_hists,), dtype=data_str)
    hist_bins = h5_file.create_dataset('hist_bins', (num_hists,), dtype=data_float32)
    hist_legend = h5_file.create_dataset('hist_legend', (num_hists,), dtype=data_vlen_str)

    hist_passed = h5_file.create_dataset('hist_passed', (num_hists,), dtype=data_int32)
    hist_total = h5_file.create_dataset('hist_total', (num_hists,), dtype=data_int32)
    hist_stack = h5_file.create_dataset('hist_stack', (num_hists, 15), dtype=data_int32)
    hist_1d = h5_file.create_dataset('hist_1d', (num_hists,), dtype=data_int32)

    for i, hist in enumerate(hist_map): # {name, type, hist}
        hist_name[i] = hist['hist_name']
        hist_type[i] = hist['type']
        hist_dict = hist['hist'].get_hist()
        hist_xlabel[i] = hist_dict['xlabel']
        hist_ylabel[i] = hist_dict['ylabel']
        hist_bins[i] = hist_dict['bins']
        hist_legend[i] = hist_dict['legend']
        if hist['type'] == 'efficiency':
            hist_total[i] = hist_dict['total']
            hist_passed[i] = hist_dict['passed']
        elif hist['type'] == 'stack':
            for j in range(len(hist_dict['hist'])):
                hist_stack[i, j] = hist_dict['hist'][j]
        elif hist['type'] == 'hist':
            hist_1d[i] = hist_dict['hist']

    h5_file.close()


def save_unfold_variables(results, beam_var, pi0_var, file_name):

    hist_map, event_mask, cut_signal_selected, cut_total_selected, signal_total, events, beam_events = results

    bool_event_mask = np.arange(len(event_mask)) == event_mask

    # Beam variables should be pi+
    print("Getting beam cross section variables!")
    beam_var_dict = beam_var.vars.get_xsec_variable(event_record=beam_events, reco_int_mask=bool_event_mask)

    # Pi0 variables should be from CeX
    print("Getting pi0 cross section variables!")
    pi0_var_dict = pi0_var.vars.get_xsec_variable(event_record=events, reco_int_mask=np.ones(len(events)).astype(bool))

    print("Saving variables to file")
    np.save(file_name, beam_var_dict)
    np.save(file_name, pi0_var_dict)


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    time.sleep(0.1)
    return event_handler_instance.run_selection(events=data)


def start_analysis(flist, config, branches):

    data = uproot.concatenate(files=flist, expressions=branches)
    return event_selection(config=config, data=data)


def configure(config_file):

    with open(config_file, "r") as cfg:
        return json.load(cfg)


def get_branches(is_mc):

    branches = ["event", "reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG", "reco_beam_passes_beam_cuts",
                "reco_daughter_PFP_true_byHits_parPDG",
                "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy", "reco_daughter_PFP_trackScore_collection",
                "reco_daughter_allTrack_Chi2_proton", "reco_daughter_allTrack_Chi2_ndof",
                "reco_daughter_PFP_michelScore_collection", "reco_beam_calo_startX",  "reco_beam_calo_startY",
                "reco_beam_calo_startZ", "reco_beam_calo_endX", "reco_beam_calo_endY", "reco_beam_calo_endZ",
                "reco_beam_true_byHits_endProcess", "reco_daughter_PFP_nHits",
                "reco_beam_vertex_michel_score", "reco_beam_vertex_nHits", "reco_daughter_allTrack_dEdX_SCE",
                "reco_beam_endX", "reco_beam_endY", "reco_beam_endZ", "reco_beam_trackEndDirX", "reco_beam_trackEndDirY",
                "reco_beam_trackEndDirZ", "dEdX_truncated_mean", "reco_beam_calo_startDirX", "reco_beam_calo_startDirY",
                "reco_beam_calo_startDirZ", "reco_beam_calo_endDirX", "reco_beam_calo_endDirY", "reco_beam_calo_endDirZ"]

    branches += ["beam_inst_C0", "beam_inst_valid", "beam_inst_trigger", "beam_inst_nMomenta", "beam_inst_nTracks", 
                 "beam_inst_TOF", "beam_inst_P", "reco_reconstructable_beam_event"]

    branches += ["fit_pi0_energy", "fit_pi0_cos_theta", "fit_pi0_gamma_energy1", "fit_pi0_gamma_energy2", "fit_pi0_gamma_oa"]

    #branches += ["reco_all_spacePts_X", "reco_all_spacePts_Y", "reco_all_spacePts_Z", "reco_all_spacePts_Integral"]

    #################
    # Truth variables
    if is_mc:
        branches += ["true_beam_daughter_startP", "true_beam_daughter_PDG", "true_daughter_nPiMinus", "true_daughter_nPiPlus",
                     "true_daughter_nPi0", "true_daughter_nProton", "true_daughter_nNeutron","true_beam_PDG",
                     "true_beam_endProcess","true_beam_endZ","true_beam_endP","true_beam_endPx", "true_beam_endPy",
                     "true_beam_endPz", "true_beam_daughter_startPx", "true_beam_daughter_startPy", "true_beam_daughter_startPz"]

        branches += ["true_beam_interactingEnergy", "true_beam_incidentEnergies", "true_beam_traj_Z", "true_beam_traj_KE"]

        branches += ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz",
                     "true_beam_Pi0_decay_PDG", "true_beam_Pi0_decay_ID", "true_beam_Pi0_decay_startP",
                     "true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ",
                     "true_beam_endX", "true_beam_endY", "true_beam_endZ"]

    return branches


if __name__ == "__main__":


    # Provide a text file with one file per line
   # files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/ana_alldaughter_files.txt"

   # file_list = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_9.root:pduneana/beamana"
    #file_list = "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset*/pduneana*.root:beamana"

    ## Official ntupeles
    #file_list = "/nfs/disk1/users/jon/pdsp_prod4a_official_ntuples/2gev/data/PDSPProd4_data_2GeV_reco2_ntuple_v09_42_03_01.root:beamana"
    #file_list = "/nfs/disk1/users/jon/pdsp_prod4a_official_ntuples/2gev/mc/PDSPProd4a_MC_2GeV_reco1_sce_datadriven_v1_ntuple_v09_41_00_03.root:beamana"
    
    #file_list = "/nfs/disk1/users/jon/custom_ntuples/data/run5429/pi0_reco/pduneana_*.root:beamana;3" 
    #file_list = "/nfs/disk1/users/jon/custom_ntuples/mc/pi0_reco/pduneana_*.root:beamana;3" 

    file_list = "/nfs/disk1/users/jon/custom_ntuples/mc/pi0_reco*/pduneana_*.root:beamana"
    #file_list = "/nfs/disk1/users/jon/custom_ntuples/data/run5429/pduneana_*.root:beamana;3"
    #file_list = "/nfs/disk1/users/jon/custom_ntuples/data/to_ana/run*/pduneana_*.root:beamana"

    cfg_file = "config/unfolding.json"
    beam_config = configure(cfg_file)
    cfg_file = "config/pi0_unfolding.json"
    pi0_config = configure(cfg_file)

    beam_unfold = Unfold(config_file=beam_config)
    pi0_unfold = Unfold(config_file=pi0_config)

    # Get main configuration
    cfg_file = "config/main.json"
    config = configure(cfg_file)

    tree_name = "pionana/beamana;17"
    branches = get_branches(is_mc=config["is_mc"])

    # Start the analysis threads
    print("Starting threads")
    start = timer()
    results = start_analysis(file_list, config, branches)

    save_unfold_variables(results, beam_var=beam_unfold, pi0_var=pi0_unfold)
    save_results(results=results, is_mc=config["is_mc"])

    end = timer()
    print("Completed Analysis! (", round((end - start), 4), "s)")

