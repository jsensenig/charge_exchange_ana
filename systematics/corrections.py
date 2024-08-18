from abc import abstractmethod
import numpy as np
import json


def get_all_corrections():
    return {cls.__name__: cls for cls in CorrectionBase.__subclasses__()}


class CorrectionBase:
    """
    Implement the corrections in subclasses.
    Meant to be applied to the events when getting the
    cross section variables.
    """
    def __init__(self, config):
        self.config = self.configure(config_file=config)

    @abstractmethod
    def apply(self, to_correct):
        """
        Implement the correction here. The cuts will pass the events.
        """
        pass

    @staticmethod
    def configure(config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        with open(config_file, "r") as cfg:
            config = json.load(cfg)

        return config


class UpstreamEnergyLoss(CorrectionBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["UpstreamEnergyLoss"]
        self.correction_var = self.local_config["correction_var"]
        # m=0.583 b=-1160
        self.slope, self.intercept = self.local_config["eloss_slope"], self.local_config["eloss_intercept"]

    def apply(self, to_correct):
        energy_loss = self.slope * to_correct + self.intercept
        return energy_loss


class BeamMomentumReweight(CorrectionBase):
    """
    The simulated beam momenta as of prod4a has variance which is underestimated.
    Apply reweighting to the simulated data to better match data.
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["BeamMomentumReweight"]

        self.mu0 = self.local_config["mc_beam_mom_mu"]
        self.sigma0 = self.local_config["mc_beam_mom_sigma"]
        self.mu = self.local_config["data_beam_mom_mu"]
        self.sigma = self.local_config["data_beam_mom_sigma"]

    def apply(self, to_correct):

        beam_mom = to_correct

        gauss_num = np.exp(((beam_mom - self.mu0) * (beam_mom - self.mu0)) / (2. * self.sigma0 ** 2))
        gauss_denom = np.exp(((beam_mom - self.mu) * (beam_mom - self.mu)) / (2. * self.sigma ** 2))

        event_weight = gauss_num / gauss_denom
        event_weight = np.clip(event_weight, a_min=1/3., a_max=3.)
        # smeared = events["beam_momentum"] + np.random.normal(loc=self.beam_mu, scale=self.beam_sigma, size=len(events))
        return event_weight


class Pi0Energy(CorrectionBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Pi0Energy"]

    def apply(self, to_correct):
        pass

    class Pi0Angle(CorrectionBase):
        """
        Uncertainty from beam energy loss upstream of the active volume.
        Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
         """

        def __init__(self, config):
            super().__init__(config=config)
            self.local_config = self.config["Pi0Angley"]

        def apply(self, to_correct):
            pass