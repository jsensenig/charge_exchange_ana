from timeit import default_timer as timer
from cex_analysis.event_handler import EventHandler
from cex_analysis.histograms import Histogram
import cex_analysis.efficiency_data as eff_data
import cex_analysis.histogram_data as hdata
import concurrent.futures
import awkward as ak
import ROOT
import uproot
import os
import time
import json


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

    f.Close()


def calculate_efficiency(selected_true, selected_total, true_count_list):
    total_eff, total_sel, total_count = eff_data.combine_efficiency(cut_efficiency_dict_list=selected_true,
                                                                    cut_total_dict_list=selected_total,
                                                                    process_true_count_list=true_count_list)

    cum_eff, purity, eff, sel = eff_data.calculate_efficiency(cut_efficiency_dict=total_eff,
                                                              cut_total_dict=total_sel,
                                                              process_true_count=total_count)

    for ceff, f, p, cut, s in zip(cum_eff, eff, purity, total_eff, sel):
        print("Cut: [\033[92m", '{:<18}'.format(cut), "\033[0m] Cumulative eff:", '{:.4f}'.format(ceff),
              " Eff:", '{:.4f}'.format(f), "Purity:", '{:.4f}'.format(p), "Selection:", s, "True/Total")


def collect_write_results(config, thread_results):
    print(thread_results)
    # Result is a tuple (<Histogram Result>, <Selection Mask>)
    # We can only get the results once as the threads finish so fill a list with the tuples
    tuple_list = [future.result() for future in thread_results]
    result_hist_list = [hists[0] for hists in tuple_list]
    result_mask_list = [masks[1] for masks in tuple_list]
    result_select_list = [eff[2] for eff in tuple_list]
    result_total_list = [eff[3] for eff in tuple_list]
    result_true_count_list = [eff[4] for eff in tuple_list]
    print("Number of thread results", len(result_hist_list))

    if len(result_hist_list) < 1:
        print("No results, just returning")
        return False

    merge_hist_maps(config, result_hist_list)
    calculate_efficiency(result_select_list, result_total_list, result_true_count_list)

    selected_events = sum([ak.sum(m, axis=0) for m in result_mask_list])
    print("Selected", selected_events, " events out of", sum(result_true_count_list), "true CEX events")


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    time.sleep(0.1)
    return event_handler_instance.run_selection(events=data)


def thread_creator(flist, config, num_workers, branches):

    # Context manager handles joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use iterations of the tree read operation to batch the data for each thread
        for i, array in enumerate(uproot.iterate(files=flist, expressions=branches, report=True, num_workers=num_workers)):
            print("---------- Starting thread", i, "----------")
            futures.append(executor.submit(event_selection, config, array[0]))
            print(array[1]) # The report part of the array tuple from the tree iterator
            time.sleep(0.2)
        collect_write_results(config, concurrent.futures.as_completed(futures))


def configure(config_file):

    with open(config_file, "r") as cfg:
        return json.load(cfg)

############################


tree_name = "pionana/beamana;17"
branches = ["reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG", "reco_beam_passes_beam_cuts",
            "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy", "reco_daughter_PFP_trackScore_collection",
            "reco_daughter_allTrack_Chi2_proton", "reco_daughter_allTrack_Chi2_ndof", "beam_inst_TOF",
            "true_daughter_nPiMinus", "true_daughter_nPiPlus", "true_daughter_nPi0", "true_daughter_nProton",
            "true_daughter_nNeutron", "true_beam_PDG", "true_beam_endProcess", "true_beam_PDG",
            "reco_daughter_PFP_michelScore_collection", "reco_beam_calo_startX", "reco_beam_calo_startY",
            "reco_beam_calo_startZ", "reco_beam_calo_endX", "reco_beam_calo_endY", "reco_beam_calo_endZ"]

# Provide a text file with one file per line
#files = "/Users/jsen/tmp/pion_qe/2gev_single_particle_sample/ana_alldaughter_files.txt"
#files = "~/tmp/pion_qe/2gev_full_mc_sample/newShower_n14580_no_alldaughter_all/all_files.txt"

# with open(files) as f:
#     file_list = f.readlines()
# file_list = [line.strip() for line in file_list]

#file_list = ["/Users/jsen/tmp/pion_qe/ana_scripts/merge_files/full_mc_merged.root"]

#file_list = ["/Users/jsen/tmp/pion_qe/pduneana_2gev_n2590.root"]
#file_list = ["~/tmp/pion_qe/pionana_Prod4_mc_1GeV_1_14_21.root"]
file_list = ["/Users/jsen/tmp/pion_qe/ana_scripts/merge_files/output.root"]

# Number of threads
num_workers = 4
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

#"cut_list": ["TOFCut", "BeamQualityCut", "APA3Cut", "MaxShowerEnergyCut", "ShowerCut", "DaughterPionCut", "MichelCut"],