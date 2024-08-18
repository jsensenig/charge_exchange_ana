from abc import abstractmethod
import numpy as np
import json


def get_all_systematics():
    return {cls.__name__: cls for cls in SystematicsBase.__subclasses__()}


class SystematicsBase:
    """
    Implement the systematics in subclasses.
    Meant to be applied to the events when getting the
    cross section variables.
    """
    def __init__(self, config):
        self.config = self.configure(config_file=config)

    @abstractmethod
    def apply(self, events):
        """
        Implement the systematic here. The cuts will pass the events.
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


class Statistical(SystematicsBase):
    """
    To estimate the statistical uncertainty on the cross-section we can
    Piosson flucuate both the response matrix R and efficiency. This is
    done by weighting event counts in each bin and the missed events.
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Statistical"]

    def apply(self, events):
        pass


class GeantCrossSection(SystematicsBase):
    """
    Implement G4Reweighting here.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["GeantCrossSection"]

    def apply(self, events):
        pass


class UpstreamEnergyLoss(SystematicsBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["UpstreamEnergyLoss"]
        # m=0.583 b=-1160
        # self.slope, self.intercept = self.local_config["eloss_slope"], self.local_config["eloss_intercept"]
        self.mu = self.local_config["eloss_mu"]
        self.sigma = self.local_config["eloss_sigma"]

    def apply(self, syst_var):
        # This will be in a corrections class
        # energy_shift = self.slope * syst_var + self.intercept
        # return syst_var + energy_shift
        return syst_var + np.random.normal(loc=self.mu, scale=self.sigma, size=len(syst_var))


class BeamMomentum(SystematicsBase):
    """
    The simulated beam momenta as of prod4a has variance which is underestimated.
    Apply a smearing to the simulated data to better match data.
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["BeamMomentum"]

        self.beam_mu = self.local_config["beam_mom_mu"]
        self.beam_sigma = self.local_config["beam_mom_sigma"]

    def apply(self, syst_var): # sample from a 2D gaussian for mu,sigma (mu0,sigma0 fixed)
        # should be a correction
        # smeared = events["beam_momentum"] + np.random.normal(loc=self.beam_mu, scale=self.beam_sigma, size=len(events))
        return syst_var + np.random.normal(loc=0, scale=5, size=len(syst_var))


class TrackLength(SystematicsBase):
    """
    Uncertainty from reco track length.
    Take the ratio of reco/true length and fit ratio resolution.
    Systematic variation is sigma of fit.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["TrackLength"]

        self.mu = self.local_config["length_mu"]
        self.sigma = self.local_config["length_sigma"]

    def apply(self, syst_var):
        return syst_var + np.random.normal(loc=self.mu, scale=self.sigma, size=len(syst_var))


class Pi0Energy(SystematicsBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Pi0Energy"]

    def apply(self, events):
        pass

    class Pi0Angle(SystematicsBase):
        """
        Uncertainty from beam energy loss upstream of the active volume.
        Potentially 4MeV uncertainty, flucuated and propagated to the xsec.
         """

        def __init__(self, config):
            super().__init__(config=config)
            self.local_config = self.config["Pi0Angley"]

        def apply(self, events):
            pass