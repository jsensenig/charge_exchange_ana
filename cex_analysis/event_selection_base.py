from abc import abstractmethod
from cex_analysis.histograms import Histogram


class EventSelectionBase(Histogram):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # Initialize the map of efficiency counters
        self.cut_eff_map = super().init_eff()

        # Initialize the map of histograms
        self.hist_map = super().init_hists()

    @abstractmethod
    def selection(self, events):
        pass

    def efficiency(self, cut, passed, value):
        self.cut_eff_map[cut].Fill(passed, value)

    @abstractmethod
    def plot_particles_base(self, events, pdg, precut):
        pass

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def get_cut_doc(self):
        pass
