from abc import abstractmethod
import json
import numpy as np
from bethe_bloch_utils import BetheBloch
from cex_analysis.plot_utils import bin_width_np, bin_centers_np


class XSecBase:

    def __init__(self, config_file):

        self.consts = {"rho": 1.1,
                       "Nav": 1.e23,
                       "mpip": 139.57,
                       "mpi0": 135.}

        # self.num_density = 2.11038725e22 # [cm^-3] x*y*z
        avogadro_constant = 6.02214076e23  # 1 / mol
        argon_molar_mass = 39.95  # g / mol
        liquid_argon_density = 1.39  # g / cm^3

        self.sigma_factor = argon_molar_mass / (avogadro_constant * liquid_argon_density)
        self.sigma_factor *= 1.e27  # Convert to milli-barn
        self.config = self.configure(config_file=config_file)

    @abstractmethod
    def calc_xsec(self, hist_dict):
        """
        API to the cross-section calculations
        """
        pass

    @abstractmethod
    def propagate_error(self):
        """
        Method to implement the error propagation
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


class XSecTotal(XSecBase):
    """
    The total cross-section so just for the incoming beam particles.
    This uses the E-slice method developed in,
    Stocker, F. Measurement of the Pion Absorption Cross-Section with the ProtoDUNE Experiment. Ph.D. Thesis,
    University of Bern, Bern, Switzerland, 2021.
    and revised to reduce bias as outlined in:
    https://arxiv.org/pdf/2312.09333 and more colloquially in,
    https://indico.fnal.gov/event/59095/contributions/263026/attachments/165472/219911/pionXS_HadAna_230329.pdf
    """
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

        self.local_config = self.config["XSecTotal"]
        self.eslices = np.asarray(self.local_config["eslices"])
        self.delta_e = bin_width_np(self.eslices)
        self.bethe_bloch = BetheBloch(mass=139.57, charge=1)

    def calc_xsec(self, hist_dict):
        """
        Input: 3 hists init KE, end KE, int KE
        """
        # Get the requisite histograms
        init_hist = hist_dict["init_hist"]
        end_hist = hist_dict["end_hist"]
        int_hist = hist_dict["int_hist"]
        inc_hist = self.calculate_incident(init_hist=init_hist, end_hist=end_hist)

        # Get dE/dx as a function fo KE for the center of each bin
        dedx = np.asarray([self.bethe_bloch.meandEdx(ke) for ke in bin_centers_np(self.eslices)])

        # The Eslice cross-section calculation
        xsec = (self.sigma_factor / self.delta_e) * dedx * np.log(inc_hist / (inc_hist - int_hist))

        return xsec

    def propagate_error(self):
        pass

    def calculate_incident(self, init_hist, end_hist):
        """
        The number incident on a slice is the total start number - the number which ended
        in all preceding bins.
        """
        inc_hist = np.zeros(len(init_hist))
        #for b in range(len(self.eslices) - 1):
        #    # inc_hist[b] = np.sum(np.flip(init_hist)[:b + 1]) - np.sum(np.flip(end_hist)[:b])
        #    inc_hist[b] = np.sum(np.flip(end_hist)[b:]) - np.sum(np.flip(init_hist)[(b + 1):])
        #    # inc_hist[b] = np.sum(init_hist[b:]) - np.sum(end_hist[(b + 1):])

        for ibin in range(len(self.eslices)-2):
            for itmp in range(0, ibin+1):
                inc_hist[ibin+1] += np.flip(init_hist)[itmp]
            for itmp in range(0, ibin):
                inc_hist[ibin+1] -= np.flip(end_hist)[itmp]

        return np.flip(inc_hist)


class XSecDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and a single daughter variable, e.g. pi0 Ke, proton angle wrt to beam
    """
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

        self.local_config = self.config["XSecDiff"]

    def calc_xsec(self, hist_dict):
        pass

    def propagate_error(self):
        pass


class XSecDoubleDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and two daughter variables, e.g. pi0 Ke, pi0 angle wrt to beam
    """
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

        self.local_config = self.config["XSecDoubleDiff"]

    def calc_xsec(self, hist_dict):
        pass

    def propagate_error(self):
        pass
