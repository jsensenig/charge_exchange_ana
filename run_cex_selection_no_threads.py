from timeit import default_timer as timer
from cex_analysis.event_handler import EventHandler
from cex_analysis.histograms import Histogram
import cex_analysis.efficiency_data as eff_data
import cex_analysis.histogram_data as hdata
import numpy as np
import uproot
import time
import json
import h5py


def open_file(file_name, tree_dir):
    try:
        file1 = uproot.open(file_name)
        tree = file1[tree_dir]
    except FileNotFoundError:
        print("Could not find file", file_name)
        return
    return tree


def merge_hist_maps(config, hist_maps):
    """
    Here we want to,
    1. Loop over each type of histogram
    2. Collect histograms from all threads for a given histogram type
    3. Add all histograms of the same name (they should be the same histogram
       just from a different thread)
    4. Write the summed histogram to file.
    :param config: Config to set up the
    :param hist_maps:
    :return:
    """
    # Set to batch mode so no windows pop up
    # ROOT.gROOT.SetBatch(True)

    # Open file to which we write results
    # f = ROOT.TFile("result_file.root", "RECREATE")
    #
    # hclass = Histogram(config)
    # # hist_maps = [[Thread 0],..., [Thread N]]
    # type_list = ["stack", "hist", "efficiency"]
    # for t in type_list:
    #     hist_type_list = []
    #     for res in hist_maps: # results from each thread
    #         hist_type_list += hdata.get_hist_type_list(res, t)
    #     if len(hist_type_list) < 1:
    #         continue
    #     # Only returns unique names as a set
    #     hist_names = hdata.get_hist_name_list(hist_type_list)
    #     for name in hist_names:
    #         hlist = hdata.get_select_hist_name_list(hist_type_list, name)
    #         merged_hist = None
    #         if t == "efficiency":
    #             merged_hist = hclass.merge_efficiency_list(hlist)
    #         elif t == "hist":
    #             merged_hist = hclass.merge_hist_list(hlist)
    #         elif t == "stack":
    #             merged_hist = hclass.merge_stack_list(hlist)
    #         else:
    #             print("Unknown histogram type! ", t)
    #         if merged_hist is not None:
    #             merged_hist.Write(t + "_" + name)
    #             c = ROOT.TCanvas()
    #             merged_hist.Draw()
    #             c.SaveAs("hist_figs/" + t + "_" + name + ".pdf")
    #
    # f.Close()


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


def save_results(results):

    hist_map, event_mask, cut_signal_selected, cut_total_selected, signal_total, events, beam_events = results

    # merge_hist_maps(config, result_hist_list)
    calculate_efficiency(cut_signal_selected, cut_total_selected, signal_total)

    *_, num_last_cut = cut_signal_selected.items()
    print("Selected", num_last_cut, " events out of", signal_total, "true CEX events")

    # Write cut efficiency and purity
    h5_file = h5py.File('test_cut_file.hdf5', 'w')
    h5_file.create_dataset('cut_names', data=list(cut_total_selected.keys()))
    h5_file.create_dataset('total_signal', data=signal_total)
    h5_file.create_dataset('cut_total', data=list(cut_total_selected.values()))
    h5_file.create_dataset('cut_signal', data=list(cut_signal_selected.values()))
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
        hist_name[i] = hist['name']
        hist_type[i] = hist['type']
        hist_xlabel[i] = hist['xlabel']
        hist_ylabel[i] = hist['ylabel']
        hist_dict = hist['hist'].get_hist()
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


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    time.sleep(0.1)
    return event_handler_instance.run_selection(events=data)


def start_analysis(flist, config, branches):

    data = uproot.concatenate(files={flist}, expressions=branches)
    results = event_selection(config=config, data=data)
    save_results(results=results)


def configure(config_file):

    with open(config_file, "r") as cfg:
        return json.load(cfg)


if __name__ == "__main__":

    tree_name = "pionana/beamana;17"
    branches = ["event", "reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG", "reco_beam_passes_beam_cuts",
                "reco_daughter_PFP_true_byHits_parPDG", "true_beam_daughter_startP", "true_beam_daughter_PDG",
                "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy", "reco_daughter_PFP_trackScore_collection",
                "reco_daughter_allTrack_Chi2_proton", "reco_daughter_allTrack_Chi2_ndof", "beam_inst_TOF",
                "true_daughter_nPiMinus", "true_daughter_nPiPlus", "true_daughter_nPi0", "true_daughter_nProton",
                "true_daughter_nNeutron","true_beam_PDG","true_beam_endProcess","true_beam_endZ","true_beam_endP","true_beam_endPx",
                "true_beam_endPy", "true_beam_endPz", "reco_daughter_PFP_michelScore_collection", "reco_beam_calo_startX",
                "reco_beam_calo_startY", "reco_beam_calo_startZ", "reco_beam_calo_endX", "reco_beam_calo_endY",
                "reco_beam_calo_endZ", "true_beam_daughter_startPx", "true_beam_daughter_startPy", "true_beam_daughter_startPz",
                "reco_beam_true_byHits_endProcess", "reco_daughter_PFP_nHits", "beam_inst_P", "reco_beam_vertex_michel_score",
                "reco_beam_vertex_nHits", "reco_daughter_allTrack_dEdX_SCE", "reco_beam_endX", "reco_beam_endY",
                "reco_beam_endZ"]

    branches += ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz",
                 "true_beam_Pi0_decay_PDG", "true_beam_Pi0_decay_ID", "true_beam_Pi0_decay_startP",
                 "true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ",
                 "true_beam_endX", "true_beam_endY", "true_beam_endZ", "reco_beam_trackEndDirX",
                 "reco_beam_trackEndDirY", "reco_beam_trackEndDirZ", "true_beam_interactingEnergy",
                 "true_beam_incidentEnergies", "true_beam_traj_Z", "true_beam_traj_KE"]#, "dEdX_truncated_mean"]

    branches += ["reco_beam_calo_startDirX", "reco_beam_calo_startDirY", "reco_beam_calo_startDirZ",
                 "reco_beam_calo_endDirX", "reco_beam_calo_endDirY", "reco_beam_calo_endDirZ",
                 "true_beam_slices"]#, "true_beam_traj_incidentEnergies", "true_beam_traj_interacting_Energy"]

    branches += ["reco_all_spacePts_X", "reco_all_spacePts_Y", "reco_all_spacePts_Z", "reco_all_spacePts_Integral"]

    # Provide a text file with one file per line
    files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/ana_alldaughter_files.txt"

    file_list = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_9.root:pduneana/beamana"
    file_list = "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana*.root:beamana"

    # Number of threads
    # num_workers = 1
    # num_workers = check_thread_count(num_workers)

    # Get main configuration
    cfg_file = "config/main.json"
    config = configure(cfg_file)

    # Start the analysis threads
    print("Starting threads")
    start = timer()
    start_analysis(file_list, config, branches)

    end = timer()
    print("Completed Analysis! (", round((end - start), 4), "s)")

