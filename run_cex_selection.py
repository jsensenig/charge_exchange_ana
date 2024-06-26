from timeit import default_timer as timer
from cross_section.cex_dd_cross_section import CexDDCrossSection
from cex_analysis.event_handler import EventHandler
from cex_analysis.histograms import Histogram
import cex_analysis.efficiency_data as eff_data
import cex_analysis.histogram_data as hdata
import concurrent.futures
import awkward as ak
import numpy as np
import ROOT
import uproot
import time
import json
import os


def check_thread_count(threads):
    if threads > os.cpu_count():
        print("Requested", threads, "threads but only", os.cpu_count(), "available!")
        print("Setting number of threads to", os.cpu_count())
        return os.cpu_count()
    return threads


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
    ROOT.gROOT.SetBatch(True)

    # Open file to which we write results
    f = ROOT.TFile("result_file.root", "RECREATE")

    hclass = Histogram(config)
    # hist_maps = [[Thread 0],..., [Thread N]]
    type_list = ["stack", "hist", "efficiency"]
    for t in type_list:
        hist_type_list = []
        for res in hist_maps: # results from each thread
            hist_type_list += hdata.get_hist_type_list(res, t)
        if len(hist_type_list) < 1:
            continue
        # Only returns unique names as a set
        hist_names = hdata.get_hist_name_list(hist_type_list)
        for name in hist_names:
            hlist = hdata.get_select_hist_name_list(hist_type_list, name)
            merged_hist = None
            if t == "efficiency":
                merged_hist = hclass.merge_efficiency_list(hlist)
            elif t == "hist":
                merged_hist = hclass.merge_hist_list(hlist)
            elif t == "stack":
                merged_hist = hclass.merge_stack_list(hlist)
            else:
                print("Unknown histogram type! ", t)
            if merged_hist is not None:
                merged_hist.Write(t + "_" + name)
                c = ROOT.TCanvas()
                merged_hist.Draw()
                c.SaveAs("hist_figs/" + t + "_" + name + ".pdf")

    f.Close()


def calculate_efficiency(selected_true, selected_total, true_count_list):
    total_eff, total_sel, total_count = eff_data.combine_efficiency(cut_efficiency_dict_list=selected_true,
                                                                    cut_total_dict_list=selected_total,
                                                                    process_true_count_list=true_count_list)

    cum_eff, purity, eff, sel, fom = eff_data.calculate_efficiency(cut_efficiency_dict=total_eff,
                                                                   cut_total_dict=total_sel,
                                                                   process_true_count=total_count)

    for ceff, f, p, cut, s, l in zip(cum_eff, eff, purity, total_eff, sel, fom):
        print("Cut: [\033[92m", '{:<18}'.format(cut), "\033[0m] Cumulative eff:", '{:.4f}'.format(ceff),
              " Eff:", '{:.4f}'.format(f), "Purity:", '{:.4f}'.format(p),
              "S/sqrt(S+B):", '{:.4f}'.format(l), "Selection:", s, "True/Total")


def collect_write_results(config, thread_results, flist, branches):
    print(thread_results)
    # Result is a tuple (<Histogram Result>, <Selection Mask>)
    # We can only get the results once as the threads finish so fill a list with the tuples
    tuple_list = [future.result() for future in thread_results]

    # 1. The histograms created in the selection
    result_hist_list = [hists[0] for hists in tuple_list]

    # 2. The selection masks created in the selection
    event_selection_mask = np.array([], dtype=bool)
    selected_events_count = 0
    for masks in tuple_list:
        selected_events_count += ak.sum(masks[1])
        event_selection_mask = np.hstack((event_selection_mask, ak.to_numpy(masks[1])))

    # 3. The cut efficiency and purity
    result_select_list = [eff[2] for eff in tuple_list]

    # 4. The selected plots created in the selection (to be used in efficiency plots)
    result_total_list = [eff[3] for eff in tuple_list]

    # 5. The true selected events (to be used in efficiency plots)
    result_true_count_list = [eff[4] for eff in tuple_list]

    # 6. The selected events
    selected_events = [eff[5] for eff in tuple_list]

    # 7. The selected beam events
    selected_beam_events = [eff[6] for eff in tuple_list]

    print("Number of thread results", len(result_hist_list))

    if len(result_hist_list) < 1:
        print("No results, just returning")
        return False

    merge_hist_maps(config, result_hist_list)
    calculate_efficiency(result_select_list, result_total_list, result_true_count_list)

    #all_events = uproot.concatenate(files=flist, expressions=branches)
    #all_events = uproot.concatenate(files={flist[0]:"beamana;121"}, expressions=branches)

    #xsec = CexDDCrossSection(None)
    #xsec.extract_cross_section(beam_events=selected_beam_events[0], selected_events=selected_events[0], total_incident_pion=14000)

    #selected_events = sum([ak.sum(m, axis=0) for m in result_mask_list])
    print("Selected", selected_events_count, " events out of", sum(result_true_count_list), "true CEX events")


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    time.sleep(0.1)
    return event_handler_instance.run_selection(events=data)


def thread_creator(flist, config, num_workers, branches):

    # Context manager handles joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use iterations of the tree read operation to batch the data for each thread
        #for i, array in enumerate(uproot.iterate(files=flist, expressions=branches, report=True, step_size='10000 MB', num_workers=num_workers)):
        ##for i, array in enumerate(uproot.iterate(files=flist, expressions=branches, report=True, num_workers=num_workers)):
        #    print("---------- Starting thread", i, "----------")
        #    futures.append(executor.submit(event_selection, config, array[0]))
        #    print(array[1])  # The report part of the array tuple from the tree iterator
        #    time.sleep(0.2)
        data = uproot.concatenate(files={flist}, expressions=branches)
        futures.append(executor.submit(event_selection, config, data))
        collect_write_results(config, concurrent.futures.as_completed(futures), flist, branches)


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

    # Space-point branches
    #branches += ["reco_daughter_PFP_shower_spacePts_X","reco_daughter_PFP_shower_spacePts_Y",
    #             "reco_daughter_PFP_shower_spacePts_Z", "reco_daughter_PFP_shower_spacePts_count",
    #             "reco_daughter_PFP_shower_spacePts_gmother_ID", "reco_daughter_PFP_shower_spacePts_mother_ID",
    #             "reco_daughter_PFP_shower_spacePts_gmother_PDG", "reco_daughter_PFP_shower_spacePts_E"]

    branches += ["true_beam_Pi0_decay_startPx", "true_beam_Pi0_decay_startPy", "true_beam_Pi0_decay_startPz",
                 "true_beam_Pi0_decay_PDG", "true_beam_Pi0_decay_ID", "true_beam_Pi0_decay_startP",
                 "true_beam_Pi0_decay_startX", "true_beam_Pi0_decay_startY", "true_beam_Pi0_decay_startZ",
                 "true_beam_endX", "true_beam_endY", "true_beam_endZ", "reco_beam_trackEndDirX",
                 "reco_beam_trackEndDirY", "reco_beam_trackEndDirZ", "true_beam_interactingEnergy",
                 "true_beam_incidentEnergies"]#, "dEdX_truncated_mean"]

    branches += ["reco_beam_calo_startDirX", "reco_beam_calo_startDirY", "reco_beam_calo_startDirZ",
                 "reco_beam_calo_endDirX", "reco_beam_calo_endDirY", "reco_beam_calo_endDirZ",
                 "true_beam_slices"]#, "true_beam_traj_incidentEnergies", "true_beam_traj_interacting_Energy"]

    branches += ["reco_all_spacePts_X", "reco_all_spacePts_Y", "reco_all_spacePts_Z", "reco_all_spacePts_Integral"]

    # Provide a text file with one file per line
    files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/ana_alldaughter_files.txt"

    #with open(files) as f:
    #    file_list = f.readlines()
    #file_list = [line.strip() for line in file_list]

    # Full MC merged file
    #file_list = ["/Users/jsen/tmp/pion_qe/ana_scripts/merge_files/output.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/pduneana_2gev_sub1_45972403_0_4_n500.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/pduneana_full_mc_n500.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/full_mc_shower_sample/pduneana_n1000.root"]
    # file_list = ["/Users/jsen/tmp/tmp_pi0_shower/full_mc_shower_sample/full_mc_lar_v09_35_00_n13500/full_mc_shower_sp_merged_n7500.root"]

    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/full_mc_shower_sample/full_mc_lar_v09_35_00_n13500/full_mc_shower_sp_merged_n13300.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_n9500/full_mc_shower_sp_noecut_merged_n9500.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_unique_sample_n3000/full_mc_shower_sp_merged_unique_3000.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_unique_sample_n14000/full_mc_shower_sp_merged_unique_n13500.root"]
    #file_list = ["/Users/jsen/tmp/tmp_pi0_shower/tmp_no_ecut_unique_sample_n100k/full_mc_sp_merged_unique_n86k.root"]
    #file_list = ["/Users/jsen/tmp/pion_qe/cex_selection/macros/merge_files/full_mc_merged_n86k_new_inc_tmean_branch.root"]
    file_list = ["/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_0.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_1.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_2.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_3.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_4.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_5.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_6.root:pduneana/beamana",
                "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset0/pduneana_7.root:pduneana/beamana"]
    file_list = "/home/jon/work/protodune/analysis/pi0_reco/data/2gev_ana_files/subset*/pduneana_*.root:pduneana/beamana"

    # Number of threads
    num_workers = 1
    num_workers = check_thread_count(num_workers)

    # Get main configuration
    cfg_file = "config/main.json"
    config = configure(cfg_file)

    # Start the analysis threads
    print("Starting threads")
    start = timer()
    thread_creator(file_list, config, num_workers, branches)

    end = timer()
    print("Completed Analysis! (", round((end - start), 4), "s)")

