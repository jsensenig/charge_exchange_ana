import cex_analysis.event_handler as eh
import concurrent.futures
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


def print_thread_result(thread_result):
    for future in thread_result:
        for f in future.result():
            try:
                print("TR", f)
            except ValueError:
                print("No thread result found")


def event_selection(config, data):
    event_handler_instance = eh.EventHandler(config)
    return event_handler_instance.run_selection(events=data)


def thread_creator(config, num_workers, tree, steps, branches):
    if num_workers > os.cpu_count():
        print("Requested", num_workers, "threads but only", os.cpu_count(), "available!")
        return
    # Context manager handles joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, array in enumerate(tree.iterate(expressions=branches, step_size=steps, report=True)):
            print("---------- Starting thread", i, "----------")
            futures.append(executor.submit(event_selection, config, array[0]))
            print(array[1])
        print_thread_result(concurrent.futures.as_completed(futures))


############################


tree_name = "pduneana/beamana;2"
# file = "/Users/jsen/tmp/pion_qe/pduneana_2gev_n2590.root"
file = "~/tmp/pion_qe/2gev_single_particle_sample/v1_all_daughter/pduneana_0.root"
branches = ["reco_daughter_PFP_true_byHits_startZ", "reco_daughter_PFP_true_byHits_PDG"]

num_workers = 2

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
