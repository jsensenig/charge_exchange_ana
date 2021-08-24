from cex_analysis.cut_factory import CutFactory
from cex_analysis.event_selection_base import EventSelectionBase
from cex_analysis.histograms import Histogram

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

        # Initialize the histogram class
        Histogram(config=config)

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
            print("Importing module: [", name, "]")
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
            else:
                print("Cut class", cut_class, "not found!")

    def test_cuts(self):
        pass

    def run_selection(self, events):
        for cut in self.cut_map:
            if cut != "TOFCut":
                continue
            print("Processing cut:", cut)
            event_mask = self.cut_map[cut].selection(events)
        # Return a subsample
        return event_mask[0:2]


