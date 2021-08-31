from dataclasses import dataclass


@dataclass
class EfficiencyData:

    num_true_process: int
    cut_efficiency_dict: dict
    cut_total_dict: dict

    """
    Class functions to set and calculate efficiency and purity
    """

    # Set the true number of events of a given process
    def set_num_true_process(self, num_true_process):
        print("True number of events", num_true_process)
        self.num_true_process = num_true_process

    # Add a dictionary with the cut name and number of events selected
    def add_cut_selection(self, cut_name, num_true_selected, num_total_selected):
        self.cut_efficiency_dict[cut_name] = num_true_selected
        self.cut_total_dict[cut_name] = num_total_selected


"""
Helper functions to set and calculate efficiency and purity
"""


def combine_efficiency(cut_efficiency_dict_list, cut_total_dict_list, process_true_count_list):

    total_true_count = process_true_count_list[0]
    total_cut_eff_dict = cut_efficiency_dict_list[0]
    total_cut_total_dict = cut_total_dict_list[0]

    for cut_eff, cut_tot, true_cnt in zip(cut_efficiency_dict_list[1:], cut_total_dict_list[1:], process_true_count_list[1:]):
        total_true_count += true_cnt
        for cut1, cut2 in zip(cut_eff, cut_tot):
            total_cut_eff_dict[cut1] += cut_eff[cut1]
            total_cut_total_dict[cut2] += cut_tot[cut2]

    return total_cut_eff_dict, total_cut_total_dict, total_true_count


def calculate_efficiency(cut_efficiency_dict, cut_total_dict, process_true_count):

    purity_list = []
    cum_eff_list = []
    eff_list = []
    selection_list = []

    for i, cut in enumerate(cut_efficiency_dict):
        if process_true_count < 1:
            print("Number of true events not set!")
            continue
        eff = cut_efficiency_dict[cut] / process_true_count
        if i == 0:
            eff_list.append(eff)
        else:
            eff_list.append(cum_eff_list[-1] - eff)
        cum_eff_list.append(eff)
        purity_list.append(cut_efficiency_dict[cut] / cut_total_dict[cut])
        selection_list.append(str(cut_efficiency_dict[cut]) + "/" + str(cut_total_dict[cut]))

    return cum_eff_list, purity_list, eff_list, selection_list


