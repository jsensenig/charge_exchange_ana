from cex_analysis.event_handler import EventHandler
from cex_analysis.histograms import Histogram
import cex_analysis.histogram_data as hdata
import concurrent.futures
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


def collect_write_results(config, thread_results):

    result_list = [future.result() for future in thread_results]
    print("Number of thread results", len(result_list))

    if len(result_list) < 1:
        print("No results, just returning")
        return False

    merge_hist_maps(config, result_list)


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    time.sleep(0.1)
    return event_handler_instance.run_selection(events=data)


def thread_creator(flist, config, num_workers, tree, steps, branches):

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


# tree_name = "pduneana/beamana;2"
tree_name = "pionana/beamana"
# file = "/Users/jsen/tmp/pion_qe/pduneana_2gev_n2590.root"
file = "~/tmp/pion_qe/pionana_Prod4_mc_1GeV_1_14_21.root"
branches = ["reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG", "reco_beam_passes_beam_cuts",
            "reco_beam_true_byHits_PDG", "reco_daughter_allShower_energy", "reco_daughter_PFP_trackScore_collection",
            "reco_beam_calo_endZ"]

file_list = ["~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_0.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_1.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_2.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_3.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_4.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_5.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_6.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_7.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_8.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_9.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_10.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_11.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_12.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_13.root",
             "~/tmp/pion_qe/2gev_single_particle_sample/v0_limited_daughter/pduneana_14.root"]

# Number of threads
num_workers = 1
num_workers = check_thread_count(num_workers)

# Get main configuration
cfg_file = "config/main.json"
config = configure(cfg_file)

# Start the analysis threads
print("Starting threads")
thread_creator(file_list, config, num_workers, "tree", "steps", branches)

print("Completed Analysis!")
