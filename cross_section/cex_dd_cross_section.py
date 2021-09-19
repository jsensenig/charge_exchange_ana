from cross_section.truth_cross_section import TruthCrossSection
from cross_section.reco_cross_section import RecoCrossSection
from ROOT import TMath, TGraph, TGraphErrors
import numpy as np
import threading
import json


class CexDDCrossSection:

    def __init__(self, config):

        self.config = config
        # Map to get the Geant cross section
        self.geant_xsec_dict = {}

    def run_truth_cross_section(self, events, event_mask):
        TruthCrossSection.extract_truth_xsec(events=events[event_mask])

    def run_reco_cross_section(self, events, event_mask):
        RecoCrossSection.extract_reco_xsec(events=events[event_mask])

    def extract_cross_section(self, events, event_mask):
        self.run_reco_cross_section(events=events, event_mask=event_mask)
        if self.config["run_truth_xsec"]:
            self.run_truth_cross_section(events=events, event_mask=event_mask)

    def calculate_double_differential_xsec(self, xsec_3d_hist, total_events):

        """
        Double differential cross section wrt angle
        X = pi0 KE
        Y = pi0 angle
        Z = pi + interaction KE
        xsec calculation: https://ir.uiowa.edu/cgi/viewcontent.cgi?article=1518&context=etd(pg54)
        """

        #std::map < std::string, TGraph * > xsec_graphs;
        xsec_graphs = {}

        NA = 6.02214076e23  # 1 / mol
        MAr = 39.95         # g / mol
        Density = 1.39      # g / cm ^ 3
        Thickness = 230.5   # 222. # cm

        # N_tgt = 39.9624 / (6.022e23) * (1.3973) = 4.7492e-23 = Ar atomic mass / (Avagadro's number * LAr density)
        n_tgt = MAr / (NA * Density * Thickness)

        # Get the number of bins in energy and angle

        energy_bins = xsec_3d_hist.GetNbinsX()
        angle_bins  = xsec_3d_hist.GetNbinsY()
        beam_bins   = xsec_3d_hist.GetNbinsZ()

        for bbins in range(1, beam_bins+1):  # beam KE

            beam_center = xsec_3d_hist.GetZaxis().GetBinCenter(bbins)
            bbin_width  = xsec_3d_hist.GetZaxis().GetBinWidth(bbins)

            for abins in range(1, angle_bins+1):  # angular xsec
                # Get the angular bin center and width
                angle_center = xsec_3d_hist.GetYaxis().GetBinCenter(abins)
                abin_width   = xsec_3d_hist.GetYaxis().GetBinWidth(abins)

                # Project out the energy so we can get the error bars
                h_xerr = xsec_3d_hist.ProjectionX("h_xerr", abins, abins, bbins, bbins, "e")
                xsec, true_xsec, true_xsec_xerr, xsec_yerr, true_xsec_yerr, energy, xsec_xerr = [], [], [], [], [], [], []

                for ebins in range(1, energy_bins+1):  # energy xsec
                    # Get the energy bin center and width
                    energy_center = xsec_3d_hist.GetXaxis().GetBinCenter(ebins)
                    ebin_width = xsec_3d_hist.GetXaxis().GetBinWidth(ebins)
                    Ni = xsec_3d_hist.GetBinContent(ebins, abins, bbins)

                    # xsec calculation
                    xsec_calc = ((Ni * n_tgt) / (total_events * ebin_width * abin_width)) * 1.e30
                    xsec.append(xsec_calc)  # [milli-barn (mb)]
                    xsec_yerr.append((xsec_calc / Ni) * h_xerr.GetBinError(ebins))

                    # True xsec
                    true_xsec.append(self.get_geant_cross_section(energy_center, angle_center, beam_center))
                    true_xsec_xerr.append(0.)
                    true_xsec_yerr.append(0.)

                    energy.append(energy_center)
                    xsec_xerr.append(ebin_width / 2.)

                    print("Ebin width ", ebin_width, " Abin width ", abin_width, " Ni ", Ni, " Energy ", energy_center,
                          " Angle ", angle_center, " Xsec ", xsec_calc)

                # Get TGraph
                gr_name = "beam_" + str(int(beam_center)) + "_angle_" + str(int(TMath.ACos(angle_center) * TMath.RadToDeg()))

                xsec_graphs[gr_name] = self.plot_cross_section(xsec, angle_center, energy, beam_center, xsec_xerr,
                                                               xsec_yerr, False)

                true_gr_name = "true_beam_" + str(int(beam_center)) + "_angle_" + \
                               str(int(TMath.ACos(angle_center) * TMath.RadToDeg()))

                xsec_graphs[true_gr_name] = self.plot_cross_section(true_xsec, angle_center, energy, beam_center,
                                                                    true_xsec_xerr, true_xsec_yerr, True)

    def plot_cross_section(self, xsec, energy, angle, beam_energy, xerr, yerr, true_xsec):

        print("Writing Xsec to file")

        xsec_graph = TGraphErrors(len(angle), np.asarray(angle), np.asarray(xsec), np.asarray(xerr), np.asarray(yerr))
        xsec_graph.SetLineWidth(1)
        xsec_graph.SetMarkerStyle(8)
        xsec_graph.SetMarkerSize(0.5)
        if true_xsec:
            xsec_graph.SetLineColor(64)
        else:
            xsec_graph.SetLineColor(46)

        title = "#pi^{+} CEX Cross-section (T_{#pi^{0}} = " + str(int(energy)) + " [MeV/c])" + \
                " (T_{#pi^{+}} = " + beam_energy << " [MeV/c])"
        xsec_graph.SetTitle(title.str().c_str())
        xsec_graph.GetXaxis().SetTitle("cos#theta_{#pi^{0}}")
        xsec_graph.GetYaxis().SetTitle("#frac{d^{2}#sigma}{dT_{#pi^{0}}d#Omega_{#pi^{0}}} [#mub/MeV/sr]")

        return xsec_graph

    def get_geant_cross_section(self, energy, angle, beam):
        """
        1) Select the correct 2D plot for the incident pion energy
        2) find the global bin number corresponding to the values of energy, angle
        """
        global_bin = self.geant_xsec_dict[beam].FindBin(energy, angle)
        """
        # 3) get the content in that bin.bin content = coss section[mb]
        """
        return self.geant_xsec_dict[beam].GetBinContent(global_bin) * 1.e3  # convert to micro-barn

    @staticmethod
    def configure(config_file, cut_name):
        """
        Implement the configuration for the concrete cut class here.
        """
        # Lock the file read just to make sure there are no conflicts
        # between threads reading the same file.
        lock = threading.Lock()
        lock.acquire()

        with open(config_file, "r") as cfg:
            tmp_config = json.load(cfg)
            local_config = tmp_config[cut_name]
            local_hist_config = tmp_config["histograms"]
        lock.release()

        return local_config, local_hist_config