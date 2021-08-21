import cex_analysis.cut_factory as cf
from cex_analysis.histograms import Histogram
from cex_analysis.event_selection_base import EventSelectionBase

from pathlib import Path
import pkgutil
import importlib


class EventHandler(Histogram):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # Initialize the map of efficiency counters
        self.cut_eff_map = super().init_eff()

        # Initialize the map of histograms
        self.hist_map = super().init_hists()

        # Create the Cut Factory to assemble the cuts
        self.build_ana = cf.CutFactory()

        # Hold the class references in these maps
        self.all_cut_classes = {}
        self.get_all_cut_classes()

        # Now assemble our cuts
        self.cut_map = {}
        self.create_analysis()

        # Testing...
        self.test_cuts()

    def get_all_cut_classes(self):
        """
        1. Get all the modules located in the package directory so we can import them
        2. Get all the subclasses of EventSelectionBase since these are the Cut classes
        :return:
        """
        # Get all available Cut classes and import them
        pkg_dir = Path(__file__).parent
        print("Importing all subclasses from:", pkg_dir)

        for (module_loader, name, ispkg) in pkgutil.iter_modules([pkg_dir]):
            print("Importing module:", name)
            importlib.import_module('.' + name, __package__)

        self.all_cut_classes = {cls.__name__: cls for cls in EventSelectionBase.__subclasses__()}

    def create_analysis(self):
        """
        Create the analysis chain by creating an instance of each Cut class
        :param config:
        :return:
        """
        for cut_class in self.config["cut_list"]:
            if cut_class in self.all_cut_classes:
                self.build_ana.register_builder(cut_class, self.all_cut_classes[cut_class])
                self.cut_map[cut_class] = self.build_ana.create(cut_class, self.config)

    def test_cuts(self):
        event = {"tof": 95, "bq": 80}
        for cut in self.cut_map:
            print("!TEST! Cut", cut, ":", self.cut_map[cut].selection(event))

    def run_selection(self, data):
        event = {"tof": 95, "bq": 80}
        event_mask = [True, False, True, False]
        for i in range(0, 1000000):
            for cut in self.cut_map:
                self.cut_map[cut].selection(event)
        return event_mask


