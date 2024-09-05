from cex_analysis.cut_factory import CutFactory
from cex_analysis.event_selection_base import EventSelectionBase
# from cex_analysis.histograms import Histogram
from cex_analysis.histograms_v2 import HistogramV2
from cex_analysis.efficiency_data import EfficiencyData
from cex_analysis.true_process import TrueProcess
import awkward as ak
import numpy as np

from pathlib import Path
import pkgutil
import importlib


class EventHandler:
    def __init__(self, config):

        self.config = config

        # Create the Cut Factory to assemble the cuts
        self.build_ana = CutFactory()

        # Hold the class references in these maps
        self.all_cut_classes = {}
        self.get_all_cut_classes()

        # Now assemble our cuts
        self.cut_map = {}
        self.create_analysis()

        # Initialize the histogram class for this instance of EventHandler
        self.Hist_object = HistogramV2(config=config)

        # Initialize the efficiency data class
        self.efficiency = EfficiencyData(num_true_process=0, cut_efficiency_dict={}, cut_total_dict={})

        self.is_mc = config["is_mc"]

    def get_all_cut_classes(self):
        """
        1. Get all the modules located in the package directory so we can import them
        2. Get all the subclasses of EventSelectionBase since these are the Cut classes
        :return:
        """
        # Get all available Cut classes and import them
        pkg_dir = str(Path(__file__).parent)
        print("Importing all subclasses from:", pkg_dir)

        for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
            print("Importing module: [\033[35m", name, "\033[0m]")
            importlib.import_module('.' + name, __package__)

        self.all_cut_classes = {cls.__name__: cls for cls in EventSelectionBase.__subclasses__()}

    def create_analysis(self):
        """
        Create the analysis chain by creating an instance of each Cut class
        :param config:
        :return:
        """
        for cut_name in self.config["cut_list"]:
            class_name = cut_name.rsplit('_')[0]
            if class_name in self.all_cut_classes:
                # If the cut class is available then register and create it from the factory
                self.build_ana.register_builder(cut_name, self.all_cut_classes[class_name])
                self.cut_map[cut_name] = self.build_ana.create(cut_name, self.config, cut_name)
            else:
                print("Cut class", class_name, "not found!")

    def full_cut_likelihood(self, s, b):
        return ak.numpy.sqrt(2. * (s + b) * ak.numpy.log(1 + s/b) - 2. * s)

    def optimize_cut(self, events, cut):

        total_true = ak.count_nonzero(events["single_charge_exchange"], axis=0)

        cut_fom_list = []
        cut_sig_bkd_list = []
        cut_full_fom_list = []
        cut_eff_times_purity_list = []
        cut_eff_times_purity2_list = []
        for point in range(0, self.cut_map[cut].num_optimizations):
            self.cut_map[cut].cut_optimization()
            selected = self.cut_map[cut].selection(events, self.Hist_object, optimizing=True)
            num_true_selected = ak.count_nonzero(events["single_charge_exchange", selected], axis=0)
            num_total_selected = ak.count_nonzero(selected, axis=0)
            eff_purity = (num_true_selected / total_true) * (num_true_selected / num_total_selected)
            eff_purity2 = eff_purity + ak.numpy.sqrt((num_true_selected / num_total_selected))
            background = num_total_selected - num_true_selected
            full_opt = self.full_cut_likelihood(num_true_selected, background)
            sig_bkgd = num_true_selected / ak.numpy.sqrt(num_total_selected - num_true_selected)
            print("S/sqrt(S+B) = [\033[92m", '{:.4f}'.format(num_true_selected / ak.numpy.sqrt(num_total_selected)),
                  " e*p = ", '{:.4f}'.format(eff_purity), " Full LLR = ", '{:.4f}'.format(full_opt),
                  " S/sqrt(B) = ", '{:.4f}'.format(sig_bkgd), "\033[0m]")
            cut_fom_list.append(num_true_selected / ak.numpy.sqrt(num_total_selected))
            cut_full_fom_list.append(full_opt)
            cut_eff_times_purity_list.append(eff_purity)
            cut_eff_times_purity2_list.append(eff_purity2)
            cut_sig_bkd_list.append(sig_bkgd)

        print(cut_full_fom_list)
        print("--------------")
        print(cut_sig_bkd_list)

    def run_selection(self, events):
        """
        Here we run the selection. This means we loop over each cut
        passing the data and receving a selection mask from it in
        sequential order.
        :param events: Array of events to be analyzed
        :return:
        """
        # Add the interaction process columns
        if self.is_mc:
            classify_process = TrueProcess()
            events = classify_process.classify_event_process(events=events)

        # Get total number of true signal events
        if self.is_mc:
            true_total_single_cex = ak.count_nonzero(events["single_charge_exchange"], axis=0)
            self.efficiency.set_num_true_process(true_total_single_cex)

        total_event_mask = np.zeros(len(events)).astype(bool)
        selection_idx = np.arange(len(events)).astype(int)
        beam_selection_idx = np.arange(len(events)).astype(int)
        total_beam_mask = np.zeros(len(events)).astype(bool)
        beam_events = None

        for i, cut in enumerate(self.cut_map):
            print("******************************************************")
            print("Processing cut: [\033[34m", cut, "\033[0m] Num Events:", len(events))

            # Perform the cut selection
            if self.cut_map[cut].optimize:
                self.optimize_cut(events, cut)
            event_mask = self.cut_map[cut].selection(events, self.Hist_object)

            # Keep track of selection efficiency
            if self.is_mc:
                num_true_selected = ak.count_nonzero(events["single_charge_exchange", event_mask], axis=0)
                num_total_selected = ak.count_nonzero(event_mask, axis=0)

                self.efficiency.add_cut_selection(cut_name=cut,
                                                  num_true_selected=num_true_selected,
                                                  num_total_selected=num_total_selected)

                print("True signal events selected:", num_true_selected, " Total events:", num_total_selected,
                      "(", true_total_single_cex, "Total True sCEX)")

            # Mask out events not selected
            events = events[event_mask]
            selection_idx = selection_idx[event_mask]

            if cut in self.config["beam_cut_list"]:
                #total_beam_mask = np.zeros(len(events)).astype(bool)
                #beam_selection_idx = np.arange(len(events)).astype(int)
                beam_selection_idx = beam_selection_idx[event_mask]
                beam_events = events


        # Global selection mask
        total_event_mask[selection_idx] = True
        total_beam_mask[beam_selection_idx] = True

        # Return all the histograms generated by the selection and the mask
        return self.Hist_object.get_hist_map(), total_event_mask, total_beam_mask, self.efficiency.cut_efficiency_dict, \
               self.efficiency.cut_total_dict, self.efficiency.num_true_process, events, beam_events
