import cex_analysis.event_handler as eh
import concurrent.futures
import numpy as np
import uproot
import ROOT as rt


def open_file(file_name, tree_dir):
    try:
        file1 = uproot.open(file_name)
        tree = file1[tree_dir]
    except FileNotFoundError:
        print("Could not find file", file_name)
        return
    return tree


def event_selection(config, data):
    event_handler_instance = eh.EventHandler(config)
    return event_handler_instance.run_selection(data=data)


def thread_creator(config, num_workers, tree, steps):
    # Context manager handles the joining of the threads
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for array in tree.iterate(expressions=branches, step_size=steps, library="np", report=True):
            futures.append(executor.submit(event_selection, config, array))

        concurrent.futures.as_completed(futures)
        for future in futures:
            for f in future.result():
                try:
                    print(f)
                except:
                    print("Exception")


############################


tree_name = "pduneana/beamana;2"
file = "/Users/jsen/tmp/pion_qe/pduneana_2gev_n2590.root"
branches = ["run", "subrun", "event"]
num_workers = 2

tree = open_file(file, tree_name)
nevts = len(tree["run"].array(library="np"))
steps = (int)(nevts / num_workers + 1)
print(steps)


# Cut list is an array so order is kept
config = {"cut_list": ["TOFCut", "BeamQualityCut"], "hist_list": ["TOF"]}
print("Trying to start the threads")
thread_creator(config, 2, tree, steps)