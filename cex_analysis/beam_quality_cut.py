from cex_analysis.event_selection_base import EventSelectionBase

"""
Cut to select qulaity beam particles and help reject beam muons
"""


class BeamQualityCut(EventSelectionBase):
    def __init__(self, config):
        super().__init__(config)

    def selection(self, event):

        event_passed = True

        if event["bq"] < 85:
            event_passed = False

        super().efficiency("BeamQualityCut", event_passed, event["bq"])

        return event_passed