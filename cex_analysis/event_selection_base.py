from abc import abstractmethod
from cex_analysis.histograms import Histogram


class EventSelectionBase(Histogram):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        # Initialize the map of efficiency counters and histograms
        self.cut_eff_map = super().init_eff()
        self.hist_map = super().init_hists()

    @abstractmethod
    def selection(self, event):
        pass

    def efficiency(self, cut, passed, value):
        self.cut_eff_map[cut].Fill(passed, value)

    def plot_particles(self, event):
        pass

    def configure(self):
        pass
