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

    @abstractmethod
    def get_correction_variable(self):
        """
        Method to access the variable
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
    Potentially 4MeV uncertainty, fluctuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["UpstreamEnergyLoss"]
        self.correction_var = self.local_config["correction_var"]
        # m=0.583 b=-1160
        self.slope, self.intercept = self.local_config["eloss_slope"], self.local_config["eloss_intercept"]
        self.p0, self.p1, self.p2 = self.local_config["eloss_p0"], self.local_config["eloss_p1"], self.local_config["eloss_p2"]

    def apply(self, to_correct):
        #energy_loss = self.slope * to_correct + self.intercept
        energy_loss = self.p2 * to_correct[self.correction_var]**2 + self.p1 * to_correct[self.correction_var] + self.p0
        return energy_loss

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class MCShiftSmearBeam(CorrectionBase):
    """
    The 2GeV/c MC has the beam_inst_P simulated incorrectly as 1GeV/c mean
    and sigma. Reweighting fixes this but it's originally too far from data
    to reweight so shift and smear it first.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["MCShiftSmearBeam"]
        self.correction_var = self.local_config["correction_var"]

        self.beam_shift = self.local_config["mc_beam_shift"] # 1.
        self.sigma = self.local_config["mc_beam_inst_sigma"] # 0.1

    def apply(self, to_correct):
        energy_smear = self.beam_shift + np.random.normal(0, self.sigma, len(to_correct))
        return energy_smear 

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class BeamMomentumReweight(CorrectionBase):
    """
    The simulated beam momenta as of prod4a has variance which is underestimated.
    Apply reweighting to the simulated data to better match data.
    """
    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["BeamMomentumReweight"]
        self.correction_var = self.local_config["correction_var"]        

        self.mu0 = self.local_config["mc_beam_mom_mu"]
        self.sigma0 = self.local_config["mc_beam_mom_sigma"]
        self.mu = self.local_config["data_beam_mom_mu"]
        self.sigma = self.local_config["data_beam_mom_sigma"]

    def apply(self, to_correct):

        beam_mom = to_correct[self.correction_var]

        gauss_num = np.exp(((beam_mom - self.mu0) * (beam_mom - self.mu0)) / (2. * self.sigma0 ** 2))
        gauss_denom = np.exp(((beam_mom - self.mu) * (beam_mom - self.mu)) / (2. * self.sigma ** 2))

        event_weight = gauss_num / gauss_denom
        event_weight = np.clip(event_weight, a_min=1/3., a_max=3.)

        return event_weight

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class MuonFracReweight(CorrectionBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, fluctuates and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["MuonFracReweight"]
        self.correction_var = self.local_config["correction_var"]
        self.muon_frac_weight = self.local_config["muon_scale"]
        self.pdg_select = -13

    def apply(self, to_correct):
        # Select the muons reweight
        muons = to_correct[self.correction_var] == self.pdg_select

        weights = np.ones(len(to_correct))
        weights[muons] *= self.muon_frac_weight

        return weights

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class BeamIncBkgdScale(CorrectionBase):
    """
    Subtract the background
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["BeamIncBkgdScale"]
        self.correction_var = self.local_config["correction_var"]

        self.scale_dict = {k: self.local_config["muon_scale"] for k in self.local_config["muon_bkgd_list"]}
        self.muon_err = self.local_config["muon_scale_error"]

    def apply(self, to_correct):

        return to_correct

    def get_bkgd_scale(self, apply_syst):

        if apply_syst:
            muon_err = np.random.normal(0, self.muon_err)
            tmp_muon = {k: self.local_config["muon_scale"] + muon_err for k in self.local_config["muon_bkgd_list"]}
            return tmp_muon

        return self.scale_dict

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class BeamSignalBkgdScale(CorrectionBase):
    """
    Subtract the background
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["BeamSignalBkgdScale"]
        self.correction_var = self.local_config["correction_var"]

        tmp_zpi0 = {k: self.local_config["zpi0_scale"] for k in self.local_config["zpi0_bkgd_list"]}
        tmp_npi0 = {k: self.local_config["npi0_scale"] for k in self.local_config["npi0_bkgd_list"]}
        self.scale_dict = {**tmp_zpi0, **tmp_npi0 }

        self.zpi0_err = self.local_config["zpi0_scale_error"]
        self.npi0_err = self.local_config["npi0_scale_error"]

    def apply(self, to_correct):

        return to_correct

    def get_bkgd_scale(self, apply_syst):

        if apply_syst:
            zpi0_err = np.random.normal(0, self.zpi0_err)
            npi0_err = np.random.normal(0, self.npi0_err)

            tmp_zpi0 = {k: self.local_config["zpi0_scale"] + zpi0_err for k in self.local_config["zpi0_bkgd_list"]}
            tmp_npi0 = {k: self.local_config["npi0_scale"] + npi0_err for k in self.local_config["npi0_bkgd_list"]}

            return {**tmp_zpi0, **tmp_npi0 }

        return self.scale_dict

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class Pi0BkgdScale(CorrectionBase):
    """
    Subtract the pi0 background
    """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Pi0BkgdScale"]
        self.correction_var = self.local_config["correction_var"]

        self.pi0_scale =self.local_config["pi0_scale"]
        self.pi0_err = self.local_config["pi0_scale_error"]

    def apply(self, to_correct):

        return to_correct

    def get_bkgd_scale(self, apply_syst):

        if apply_syst:
            return self.pi0_scale + np.random.normal(0, self.pi0_err)

        return self.pi0_scale

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class Pi0Energy(CorrectionBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, fluctuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Pi0Energy"]

    def apply(self, to_correct):
        pass

    def get_correction_variable(self):
        return self.local_config["correction_var"]


class Pi0Angle(CorrectionBase):
    """
    Uncertainty from beam energy loss upstream of the active volume.
    Potentially 4MeV uncertainty, fluctuated and propagated to the xsec.
     """

    def __init__(self, config):
        super().__init__(config=config)
        self.local_config = self.config["Pi0Angle"]

    def apply(self, to_correct):
        pass

    def get_correction_variable(self):
        return self.local_config["correction_var"]
