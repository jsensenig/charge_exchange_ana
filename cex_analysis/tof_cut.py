import numpy as np
from ROOT import TEfficiency
from abc import abstractmethod
from cex_analysis.event_selection_base import EventSelectionBase


class TOFCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

    def selection(self, event):

        event_passed = True

        if event["tof"] < 85:
            event_passed = False

        super().efficiency("TOFCut", event_passed, event["tof"])

        return event_passed

