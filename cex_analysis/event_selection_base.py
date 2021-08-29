from abc import abstractmethod
import threading
import json


class EventSelectionBase:
    """
    Base class to define the form of the concrete cut classes.
    """
    def __init__(self, config):

        self.config = config

    @abstractmethod
    def selection(self, events, hists):
        """
        Main function called from the EventHandler classto perform the
        function. Call the additional class specific functions from this function.
        :param hists: Histogram class object
        :param events: Batch of events to be processed
        :return: bool array of selected events
        """
        pass

    @abstractmethod
    def efficiency(self, total_events, passed_events, cut, hists):
        """
        Keep track of the efficiency of the cut wrt to some variable `value`
        :param hists: Histogram class object
        :param total_events: Array The events variable on which the cut is applied.
        :param passed_events: bool array Array of events which passed the cut.
        :param cut: str Name of the cut
        :return:
        """
        pass

    @abstractmethod
    def plot_particles_base(self, events, pdg, precut, hists):
        """
        Take in the pre/post event selection. This is the primary way to
        keep track of how the cuts affect the data.
        :param hists: Histogram class object
        :param events: Array The batch of events pre/post cut.
        :param pdg: Array Truth particle PDG allows the plotting of particles by PDG
        :param precut: Bool Specify if this is pre or post cut
        :return:
        """
        pass

    @staticmethod
    def configure(config_file, cut_name):
        """
        Implement the configuration for the concrete cut class here.
        """
        lock = threading.Lock()
        lock.acquire()

        with open(config_file, "r") as cfg:
            tmp_config = json.load(cfg)
            local_config = tmp_config[cut_name]
            local_hist_config = tmp_config["histograms"]
        lock.release()

        return local_config, local_hist_config

    @abstractmethod
    def get_cut_doc(self):
        """
         Implement here a short description of the cut.
         A list of it's inputs and goals for the cut.
        """
        pass
