from cex_analysis.event_handler import EventHandler
from cex_analysis.histograms import Histogram
import cex_analysis.histogram_data as hdata
import concurrent.futures
import ROOT
import uproot
import os


def open_file(file_name, tree_dir):
    try:
        file1 = uproot.open(file_name)
        tree = file1[tree_dir]
    except FileNotFoundError:
        print("Could not find file", file_name)
        return
    return tree


def merge_hist_maps(config, hist_maps):
    hclass = Histogram(config)
    # hist_maps = [[Thread 0],..., [Thread N]]
    hist_names = hdata.get_hist_name_list(hist_maps[0])
    type_list = ["stack", "hist", "efficiency"]
    for t in type_list:
        hist_type_list = []
        for res in hist_maps: # results from each thread
            hist_type_list += hdata.get_hist_type_list(res, t)
        for name in hist_names:
            hlist = hdata.get_select_hist_name_list(hist_type_list, name)
            print(name, " ", hlist)
            print(hlist[0].histogram)
            print(hlist[0].histogram)
            merged_hist = hclass.sum_hist_list(hlist)
            print("MERGED_HIST", merged_hist)
            if not None or not {}:
                merged_hist.Write()



def collect_write_results(config, thread_results):

    result_list = [future.result() for future in thread_results]

    if len(result_list) < 1:
        print("No results, just returning")
        return False

    # Open file to which we write results
    ROOT.TFile.Open("result_file.root", "RECREATE")

    merge_hist_maps(config, result_list)

    return


def event_selection(config, data):
    event_handler_instance = EventHandler(config)
    return event_handler_instance.run_selection(events=data)


def thread_creator(config, num_workers, tree, steps, branches):
    if num_workers > os.cpu_count():
        print("Requested", num_workers, "threads but only", os.cpu_count(), "available!")
        print("Setting number of threads to", os.cpu_count())
        num_workers = os.cpu_count()

    # Context manager handles joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use iterations of the tree read operation to batch the data for each thread
        for i, array in enumerate(tree.iterate(expressions=branches, step_size=steps, report=True)):
            print("---------- Starting thread", i, "----------")
            futures.append(executor.submit(event_selection, config, array[0]))
            print(array[1]) # The report part of the array tuple from the tree iterator
        collect_write_results(config, concurrent.futures.as_completed(futures))


############################


tree_name = "pduneana/beamana;2"
# file = "/Users/jsen/tmp/pion_qe/pduneana_2gev_n2590.root"
file = "~/tmp/pion_qe/2gev_single_particle_sample/v1_all_daughter/pduneana_0.root"
branches = ["reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG"]

# Number of threads
num_workers = 1

tree = open_file(file, tree_name)

steps = int(len(tree["run"].array()) / num_workers) + 1
print("Data steps", steps)


# Cut list is an array so order is kept
config = {"cut_list": ["TOFCut", "BeamQualityCut"],
          "hist_list": ["TOF"],
          "reco_daughter_pdg": "reco_daughter_PFP_true_byHits_PDG",
          "TOFCut": {"cut_variable": "reco_daughter_PFP_true_byHits_startZ", "upper": 223, "lower": 10},
          "cut_plots": {"TOFCut": ["tof_cut", "TOFCut;TOF [ns];Count", 50, 150, 50]},
          "stack_pdg_list": [11, 13, 22, 111, 211, 321, 2212]}

# Start the analysis threads
thread_creator(config, num_workers, tree, steps, branches)
