from abc import abstractmethod
import json
import numpy as np
import uproot
import ROOT
import matplotlib.pyplot as plt
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
        self.eslice_edges = np.asarray(self.config["eslice_edges"])
        self.delta_e = bin_width_np(self.eslice_edges)

        self.bethe_bloch = BetheBloch(mass=139.57, charge=1)

    @abstractmethod
    def calc_xsec(self, hist_dict, beam_eslice_edges=None):
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

    def calculate_incident(self, init_hist, end_hist):
        """
        The number incident on a slice is the total start number - the number which ended
        in all preceding bins.
        """
        inc_hist = np.zeros(len(init_hist))
        for ibin in range(len(self.eslice_edges)-1):
            for itmp in range(ibin, len(init_hist)):
                inc_hist[ibin] += init_hist[itmp]
            for itmp in range(ibin+1, len(init_hist)):
                inc_hist[ibin] -= end_hist[itmp]

        return inc_hist

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
        self.geant_total_xsec = {}

    def calc_xsec(self, hist_dict, beam_eslice_edges=None):
        """
        Input: 3 hists init KE, end KE, int KE
        """
        # Get the requisite histograms
        init_hist = hist_dict["init_hist"]
        end_hist = hist_dict["end_hist"]
        int_hist = hist_dict["int_hist"]

        inc_hist = self.calculate_incident(init_hist=init_hist, end_hist=end_hist)

        # Get dE/dx as a function fo KE for the center of each bin
        dedx = np.asarray([self.bethe_bloch.meandEdx(ke) for ke in bin_centers_np(self.eslice_edges)])

        # The Eslice cross-section calculation
        xsec = int_hist * (self.sigma_factor / ( end_hist * self.delta_e)) * dedx * np.log(inc_hist / (inc_hist - end_hist))

        return xsec

    def propagate_error(self, inc_hist, end_hist, int_hist, prefactor, cov_with_inc, bin_list):
        """
        Propagate the errors through the cross-section calculation
        3 derivatives wrt Ninc, Nend and Nint
        """
        inc_minus_end = inc_hist - end_hist

        deriv_int_hist = prefactor * (1. / end_hist) * np.log(inc_hist / inc_minus_end)
        deriv_end_hist = prefactor * (int_hist / end_hist) * ((1. / inc_minus_end) - (1. / end_hist) * np.log(inc_hist / inc_minus_end))
        deriv_inc_hist = prefactor * int_hist / inc_hist / (inc_hist - end_hist)

        bin_lens = np.ma.count(bin_list, axis=1) - 3
        nbins = bin_lens[0]
        jacobian = np.zeros([nbins, 3 * nbins])

        idx = np.arange(nbins)
        jacobian[idx, idx] = deriv_inc_hist  # ∂σ/∂Ninc
        jacobian[idx, idx + nbins] = deriv_end_hist  # ∂σ/∂Nend
        jacobian[idx, idx + nbins + nbins] = deriv_int_hist  # ∂σ/∂Nint_ex

        unfolded_xsec_cov = (jacobian @ cov_with_inc) @ jacobian.T
        xsec_yerr = np.sqrt(np.diagonal(unfolded_xsec_cov))

        return unfolded_xsec_cov, xsec_yerr

    def load_geant_total_xsec(self, xsec_file):

        with uproot.open(xsec_file) as file:
            loaded_xsec = file

        for graph in loaded_xsec:
            self.geant_total_xsec[graph] = loaded_xsec[graph].values()

    def plot_beam_xsec(self, unfold_hist, yerr, process, bin_array, xlim, ylim, xsec_file):

        proc_name = {"cex": "cex_KE;1", "abs": "abs_KE;1", "inel": "total_inel_KE;1"}

        # Load Geant cross-section model if not already loaded
        if len(self.geant_total_xsec) == 0:
            self.load_geant_total_xsec(xsec_file=xsec_file)

        # Calculate the cross-section
        xsec_hists = {"init_hist": unfold_hist.sum(axis=2).sum(axis=1)[1:-1],
                      "end_hist": unfold_hist.sum(axis=2).sum(axis=0)[1:-1],
                      "int_hist": unfold_hist.sum(axis=1).sum(axis=0)[1:-1]}

        total_xsec = self.calc_xsec(hist_dict=xsec_hists)

        plt.figure(figsize=(12, 5))
        plt.errorbar(bin_centers_np(bin_array), total_xsec, yerr, bin_width_np(bin_centers_np(bin_array)) / 2,
                     capsize=2, marker='s', markersize=3, linestyle='None', color='black', label='MC Unfolded')
        plt.plot(self.geant_total_xsec[proc_name[process]][0], self.geant_total_xsec[proc_name[process]][1],
                 linestyle='--', color='indianred', label='Geant $\\sigma$')
        plt.xlabel("$T_{\\pi^+}$ [MeV]", fontsize=14)
        plt.ylabel("$\sigma$ [mb]", fontsize=14)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(np.arange(xlim[0], xlim[1]+1, 100))
        plt.legend()
        plt.show()


class XSecDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and a single daughter variable, e.g. pi0 Ke, proton angle wrt to beam
    (dsigma / dX)_i = (1/ndE)_i * (dE/dx)_i * (1 / Delta X)_j * (N^{i,j}_int / N^{i}_inc)
    i = beam particle energy bin
    j = daughter variable bin, e.g. E_{pi^0} or cos theta_{pi^0}
    Note: N^{i,j}_int is a 2D histogram of the beam particle energy and daughter variable
    """
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

        self.local_config = self.config["XSecDiff"]
        self.geant_diff_xsec = {}

    def calc_xsec(self, hist_dict, beam_eslice_edges=None):
        # Get the requisite histograms
        inc_hist = hist_dict["inc_hist"]
        int_hist = hist_dict["int_hist"]

        # Get dE/dx as a function fo KE for the center of each bin
        dedx = np.asarray([self.bethe_bloch.meandEdx(ke) for ke in bin_centers_np(beam_eslice_edges)])

        # The Eslice cross-section calculation
        total_xsec_prefactor = (self.sigma_factor / bin_width_np(beam_eslice_edges)) * dedx

        # The cross section result is going to be 1D, the same shape as the interacting histogram
        xsec_array = np.zeros_like(int_hist)

        # Loop over the y-axis of int hist, assumed to be dX
        for j in range(int_hist.shape[0]):
            xsec_array[j] = total_xsec_prefactor * (1. / bin_width_np(self.eslice_edges)) * (int_hist[j] / inc_hist)

        return xsec_array

    def propagate_error(self, inc_hist, int_hist, prefactor, cov_with_inc, bin_list):

        deriv_int_hist = prefactor * (1. / inc_hist)
        deriv_inc_hist = - prefactor * (int_hist / (inc_hist*inc_hist))

        bin_lens = np.ma.count(bin_list, axis=1) - 3
        nbins = bin_lens[0]
        jacobian = np.zeros([nbins, 2 * nbins])

        block_lower_left = cov_with_inc[:nbins, :nbins]
        block_lower_right = cov_with_inc[:nbins, 2*nbins:]
        block_upper_left = cov_with_inc[2*nbins:, :nbins]
        block_upper_right = cov_with_inc[2*nbins:, 2*nbins:]

        combined_cov = np.block([[block_lower_left, block_lower_right], [block_upper_left, block_upper_right]])

        idx = np.arange(nbins)
        jacobian[idx, idx] = deriv_inc_hist  # ∂σ/∂Ninc
        jacobian[idx, idx + nbins] = deriv_int_hist  # ∂σ/∂Nend

        unfolded_xsec_cov = (jacobian @ combined_cov) @ jacobian.T
        xsec_yerr = np.sqrt(np.diagonal(unfolded_xsec_cov))

        return unfolded_xsec_cov, xsec_yerr

    def load_geant_total_xsec(self, xsec_file):

        with uproot.open(xsec_file) as file:
            loaded_xsec = file

        for graph in loaded_xsec:
            try:
                self.geant_diff_xsec[graph] = loaded_xsec[graph].values()
            except:
                print("Could not load", graph)

    def get_geant_diff_xsec(self, pi0_var):
        if pi0_var == 'pi0_ke':
            y = self.geant_diff_xsec['inel_cex_1dKEpi0775_MeV;1']
            x = np.linspace(0, 1200, len(self.geant_diff_xsec['inel_cex_1dKEpi0775_MeV;1']))
        elif pi0_var == 'pi0_cos':
            y = self.geant_diff_xsec['inel_cex_1dcosThetapi0775_MeV;1']
            x = np.linspace(-1, 1, len(self.geant_diff_xsec['inel_cex_1dcosThetapi0775_MeV;1']))
        else:
            print("Unknown cross-section", pi0_var, "choose ['pi0_ke', 'pi0_cos']")
            raise ValueError

        return x, y

    def plot_pi0_xsec(self, unfold_hist, unfold_err, inc_hist, beam_eslices, diff_var, bin_array, xlim, xsec_file):

        if len(self.geant_diff_xsec) == 0:
            self.load_geant_total_xsec(xsec_file=xsec_file)

        xsec_x, xsec_y = self.get_geant_diff_xsec(pi0_var=diff_var)
        self.eslice_edges = bin_array

        plt.figure(figsize=(12, 5))
        if diff_var == "pi0_ke":
            xsec_hist2d = {"inc_hist": inc_hist, "int_hist": unfold_hist.sum(axis=1)}
            diff_xsec = self.calc_xsec(hist_dict=xsec_hist2d, beam_eslice_edges=beam_eslices)
            plt.errorbar(abs(bin_centers_np(bin_array)), diff_xsec, np.sqrt(unfold_err.sum(axis=1)),
                         abs(bin_width_np(bin_array)) / 2, capsize=2, marker='s', markersize=3, linestyle='None',
                         color='black', label='MC Unfolded')

            plt.plot(xsec_x, xsec_y, linestyle='--', color='indianred', label='Geant $d\\sigma / dT_{\\pi^0}$')
            plt.xlabel("$T_{\\pi^0}$ [MeV]", fontsize=14)
            plt.ylabel("$\\frac{d\\sigma}{dT_{\\pi^0}}$ [mb]")
            plt.ylim(0, 0.3)
            plt.xticks(np.arange(xlim[0], xlim[1]+1, 100))
        elif diff_var == "pi0_cos":
            xsec_hist2d = {"inc_hist": inc_hist, "int_hist": unfold_hist.sum(axis=0)}
            diff_xsec = self.calc_xsec(hist_dict=xsec_hist2d, beam_eslice_edges=beam_eslices)
            plt.errorbar(bin_centers_np(bin_array), diff_xsec, np.sqrt(unfold_err.sum(axis=0)),
                         bin_width_np(bin_array) / 2, capsize=2, marker='s', markersize=3, linestyle='None',
                         color='black', label='MC Unfolded')
            plt.plot(xsec_x, xsec_y, linestyle='--', color='indianred', label='Geant $d\\sigma / dcos\\theta_{\\pi^0}$')
            plt.xlabel("$cos\\theta_{\\pi^0}$", fontsize=14)
            plt.ylabel("$\\frac{d\\sigma}{dcos\\theta_{\\pi^0}}$ [mb]")
            plt.ylim(0, 150)
            plt.xticks(np.arange(-1, 1.1, 2 / bin_array.shape[0]))
        else:
            print("Unknown cross-section use ['pi0_ke', 'pi0_cos']")
            raise ValueError

        plt.xlim(xlim)
        plt.legend()
        plt.show()


class XSecDoubleDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and two daughter variables, e.g. pi0 Ke, pi0 angle wrt to beam
    (d^2sigma / dX dY)_i = (1/ndE)_i * (dE/dx)_i * (1 / Delta X)_j * (1 / Delta Y)_k * (N^{i,j,k}_int / N^{i}_inc)
    i = beam particle energy bin
    j = daughter variable bin, e.g. E_{pi^0} or cos theta_{pi^0}
    k = daughter variable bin, e.g. E_{pi^0} or cos theta_{pi^0}
    Note: N^{i,j,k}_int is a 3D histogram of the beam particle energy and 2 daughter variables
    """
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

        self.local_config = self.config["XSecDoubleDiff"]
        self.geant_xsec_dict = {}

    def calc_xsec(self, hist_dict, beam_eslice_edges=None):
        # Get the requisite histograms
        inc_hist = hist_dict["inc_hist"]
        int_hist = hist_dict["int_hist"]

        # Get dE/dx as a function fo KE for the center of each bin
        dedx = np.asarray([self.bethe_bloch.meandEdx(ke) for ke in bin_centers_np(beam_eslice_edges)])

        # The Eslice cross-section calculation factors
        total_xsec_prefactor = (self.sigma_factor / bin_width_np(beam_eslice_edges)) * dedx

        # The cross section result is going to be 2D, the same shape as the interacting histogram
        xsec_array = np.zeros_like(int_hist)

        # Loop over the y-axis of int hist, assumed to be dXdY
        for j in range(int_hist.shape[0]):
            for k in range(int_hist.shape[1]):
                bin_widths = 1. / (bin_width_np(self.eslice_edges[0]) * bin_width_np(self.eslice_edges[1]))
                xsec_array[j, k] = total_xsec_prefactor * bin_widths * (int_hist[j, k] / inc_hist)

        return xsec_array

    def propagate_error(self, inc_hist, int_hist, prefactor, cov_with_inc, bin_list):

        deriv_int_hist = prefactor * (1. / inc_hist)
        deriv_inc_hist = - prefactor * (int_hist / (inc_hist * inc_hist))

        bin_lens = np.ma.count(bin_list, axis=1) - 3
        nbins = bin_lens[0]
        jacobian = np.zeros([nbins, 2 * nbins])

        block_lower_left = cov_with_inc[:nbins, :nbins]
        block_lower_right = cov_with_inc[:nbins, 2 * nbins:]
        block_upper_left = cov_with_inc[2 * nbins:, :nbins]
        block_upper_right = cov_with_inc[2 * nbins:, 2 * nbins:]

        combined_cov = np.block([[block_lower_left, block_lower_right], [block_upper_left, block_upper_right]])

        idx = np.arange(nbins)
        jacobian[idx, idx] = deriv_inc_hist  # ∂σ/∂Ninc
        jacobian[idx, idx + nbins] = deriv_int_hist  # ∂σ/∂Nend

        unfolded_xsec_cov = (jacobian @ combined_cov) @ jacobian.T
        xsec_yerr = np.sqrt(np.diagonal(unfolded_xsec_cov))

        return unfolded_xsec_cov, xsec_yerr

    def get_geant_cross_section(self, energy, angle, beam):
        # If the beam KE does not exist we should know, throw error
        if int(beam) not in self.geant_xsec_dict.keys():
            print("No match for beam", int(beam), "in dictionary", self.geant_xsec_dict.keys())
            #raise RuntimeError
            return 1.

        """
        1) Select the correct 2D plot for the incident pion energy
        2) find the global bin number corresponding to the values of energy, angle
        """
        global_bin = self.geant_xsec_dict[int(beam)].FindBin(energy, angle)
        """
        # 3) get the content in that bin.bin content = cross section[mb]
        """
        return self.geant_xsec_dict[beam].GetBinContent(global_bin) * 10. #* 1.e3  # convert to micro-barn

    def load_geant_double_diff_xsec(self):
         beam_bins = np.array([1000., 1500., 1800., 2100.])
         #beam_bins = np.array([950.,1050.,1150.,1250.,1350.,1450.,1550.,1650.,1750.,1850.,1950.,2050])
         beam_energy_hist = ROOT.TH1D("beam_energy", "Beam Pi+ Kinetic Energy;T_{#pi^{+}} [MeV/c];Count", len(beam_bins)-1, beam_bins)


         geant_file = "/Users/jsen/geant_xsec/cross_section_out_1m_3ke_new_cex_p150_picut_xnucleon.root"
         geant_xsec_file = ROOT.TFile(geant_file)

         # cd into the ROOT directory so we can clone the histograms from file
         # otherwise we lose the histograms when the file is closed
         ROOT.gROOT.cd()
         for bin_i in range(1, beam_energy_hist.GetXaxis().GetNbins()+1):
             bin_center = beam_energy_hist.GetBinCenter(bin_i)
             geant_graph_name = "inel_cex_" + str(int(bin_center)) + "_MeV"
             print("Loading GEANT Xsec TGraph", geant_graph_name)
             self.geant_xsec_dict[bin_center] = geant_xsec_file.Get(geant_graph_name).Clone()

         geant_xsec_file.Close()

         for k in self.geant_xsec_dict:
             print("Loaded XSec  [ Beam KE=", k, "  Type=", type(self.geant_xsec_dict[k]), "]")
