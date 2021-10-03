from cross_section.truth_cross_section import TruthCrossSection
from cross_section.reco_cross_section import RecoCrossSection
from cex_analysis.true_process import TrueProcess
from ROOT import TMath, TGraph, TGraphErrors, TFile, TH1D, TH2D, gROOT
import numpy as np
import json


class CexDDCrossSection:

    def __init__(self, config):

        # Map to get the Geant cross section
        self.geant_xsec_dict = {}

        self.config = config
        self.configure(config_file="")

        # File to write the cross sections
        self.outfile = None

    def extract_cross_section(self, all_events, selected_events, total_incident_pion):
        """
        Calculate the cross section from the selected events
        Main callable function to the class
        :param events: Array of all events
        :param event_mask: Boolean array of selected events
        :param total_incident_pion: Total number of beam pions incident on fiducial volume
        :return:
        """
        # Start by opening the file
        self.open_file()

        # FIXME test truth cross section first!
        # Now extract and plot cross section
        #reco_3d_hist = self.run_reco_cross_section(events=events, event_mask=event_mask)
        #self.calculate_double_differential_xsec(reco_3d_hist, total_incident_pion, False)
        true_scex_mask = TrueProcess.single_charge_exchange(events=all_events)

        #if self.config["run_truth_xsec"]:
        if True:
            truth_3d_hist = self.run_truth_cross_section(events=all_events[true_scex_mask])
            self.calculate_double_differential_xsec(truth_3d_hist, total_incident_pion, True)

        # Close file and return
        self.close_file()
        return

    def open_file(self):
        ofile = "xsec_file.root"
        self.outfile = TFile(ofile, "RECREATE")
        if self.outfile is None or not self.outfile.IsOpen():
            print("File", ofile, "not open!")
            return
        print("Opened file", ofile)

    def close_file(self):
        self.outfile.Close()
        if self.outfile.IsOpen():
            print("Failed to close file", self.outfile, " :(")

    def run_truth_cross_section(self, events):
        truth_xsec = TruthCrossSection(self.config)
        return truth_xsec.extract_truth_xsec(events=events)

    def run_reco_cross_section(self, events):
        reco_xsec = RecoCrossSection(self.config)
        return reco_xsec.extract_reco_xsec(events=events)

    def calculate_double_differential_xsec(self, xsec_3d_hist, total_events, truth_vars):
        """
        Double differential cross section wrt angle
        X = pi0 KE
        Y = pi0 angle
        Z = pi + interaction KE
        xsec calculation: https://ir.uiowa.edu/cgi/viewcontent.cgi?article=1518&context=etd(pg54)
        """
        xsec_graphs = {}

        avogadro_constant    = 6.02214076e23  # 1 / mol
        argon_molar_mass     = 39.95          # g / mol
        liquid_argon_density = 1.39           # g / cm ^ 3
        fiducial_thickness   = 230.5          # 222. # cm

        # N_tgt = 39.9624 / (6.022e23) * (1.3973) = 4.7492e-23 = Ar atomic mass / (Avagadro's number * LAr density)
        num_target = argon_molar_mass / (avogadro_constant * liquid_argon_density * fiducial_thickness)

        # Get the number of bins in energy and angle
        energy_bins = xsec_3d_hist.GetNbinsX()
        angle_bins  = xsec_3d_hist.GetNbinsY()
        beam_bins   = xsec_3d_hist.GetNbinsZ()

        for beambin_i in range(1, beam_bins+1):  # beam KE

            beam_bin_center = xsec_3d_hist.GetZaxis().GetBinCenter(beambin_i)

            energy_xsec = self.plot_xsec_energy(xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                                                total_events, xsec_graphs, truth_vars)
            self.write_graphs_to_file(energy_xsec)
            angle_xsec = self.plot_xsec_angle(xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                                              total_events, xsec_graphs, truth_vars)
            self.write_graphs_to_file(angle_xsec)

    @staticmethod
    def write_graphs_to_file(graph_dict):
        # Write TGraphs to file
        for key in graph_dict:
            graph_dict[key].Write(key)

    def plot_xsec_energy(self, xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                         total_events, xsec_graphs, truth_vars):

        for anglebin_i in range(1, angle_bins+1):  # angular xsec
            # Get the angular bin center and width
            angle_center = xsec_3d_hist.GetYaxis().GetBinCenter(anglebin_i)
            abin_width   = xsec_3d_hist.GetYaxis().GetBinWidth(anglebin_i)

            # Project out the energy so we can get the error bars
            h_xerr = xsec_3d_hist.ProjectionX("h_xerr", anglebin_i, anglebin_i, beambin_i, beambin_i, "e")
            xsec, true_xsec, true_xsec_xerr, xsec_yerr, true_xsec_yerr, energy, xsec_xerr = [], [], [], [], [], [], []

            for energybin_i in range(1, energy_bins+1):  # energy xsec
                # Get the energy bin center and width
                energy_center = xsec_3d_hist.GetXaxis().GetBinCenter(energybin_i)
                ebin_width = xsec_3d_hist.GetXaxis().GetBinWidth(energybin_i)
                num_interactions = xsec_3d_hist.GetBinContent(energybin_i, anglebin_i, beambin_i)

                # xsec calculation
                xsec_calc = ((num_interactions * num_target) / (total_events * ebin_width * abin_width)) * 1.e30
                xsec.append(xsec_calc)  # [micro-barn (ub)]
                if num_interactions == 0:
                    xsec_yerr.append(0.)
                else:
                    xsec_yerr.append((xsec_calc / num_interactions) * h_xerr.GetBinError(energybin_i))

                # True xsec
                true_xsec.append(self.get_geant_cross_section(energy_center, angle_center, beam_bin_center))
                true_xsec_xerr.append(0.)
                true_xsec_yerr.append(0.)

                energy.append(energy_center)
                xsec_xerr.append(ebin_width / 2.)

                print("Ebin width ", ebin_width, " Abin width ", abin_width, " N_int ", num_interactions, " Energy ",
                      energy_center, " Angle ", angle_center, " Xsec ", xsec_calc)

            # Get TGraph
            gr_name = "beam_" + str(int(beam_bin_center)) + "_angle_" + str(int(TMath.ACos(angle_center) * TMath.RadToDeg()))
            if truth_vars:
                gr_name += "truth_vars"

            xsec_graphs[gr_name] = self.plot_cross_section(xsec, angle_center, energy, beam_bin_center, xsec_xerr,
                                                           xsec_yerr, False)

            true_gr_name = "true_beam_" + str(int(beam_bin_center)) + "_angle_" + \
                           str(int(TMath.ACos(angle_center) * TMath.RadToDeg()))

            xsec_graphs[true_gr_name] = self.plot_cross_section(true_xsec, angle_center, energy, beam_bin_center,
                                                                true_xsec_xerr, true_xsec_yerr, True)

        return xsec_graphs

    def plot_xsec_angle(self, xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                        total_events, xsec_graphs, truth_vars):

        for energybin_i in range(1, energy_bins):  # energy xsec
            # Get the energy bin center and width
            energy_center = xsec_3d_hist.GetXaxis().GetBinCenter(energybin_i)
            ebin_width = xsec_3d_hist.GetXaxis().GetBinWidth(energybin_i)

            # Project out the energy so we can get the error bars
            h_xerr = xsec_3d_hist.ProjectionX("h_xerr", energybin_i, energybin_i, beambin_i, beambin_i, "e")
            xsec, true_xsec, true_xsec_xerr, xsec_yerr, true_xsec_yerr, angle, xsec_xerr = [], [], [], [], [], [], []

            for anglebin_i in range(1, angle_bins+1):  # angular xsec
                # Get the angular bin center and width
                angle_center = xsec_3d_hist.GetYaxis().GetBinCenter(anglebin_i)
                abin_width = xsec_3d_hist.GetYaxis().GetBinWidth(anglebin_i)
                num_interactions = xsec_3d_hist.GetBinContent(energybin_i, anglebin_i, beambin_i)

                # xsec calculation
                xsec_calc = ((num_interactions * num_target) / (total_events * ebin_width * abin_width)) * 1.e30
                xsec.append(xsec_calc)  # [micro-barn (ub)]
                if num_interactions == 0:
                    xsec_yerr.append(0.)
                else:
                    xsec_yerr.append((xsec_calc / num_interactions) * h_xerr.GetBinError(anglebin_i))

                # True xsec
                true_xsec.append(self.get_geant_cross_section(energy_center, angle_center, beam_bin_center))
                true_xsec_xerr.append(0.)
                true_xsec_yerr.append(0.)

                angle.append(energy_center)
                xsec_xerr.append(ebin_width / 2.)

                print("Ebin width ", ebin_width, " Abin width ", abin_width, " N_int ", num_interactions, " Energy ",
                      energy_center, " Angle ", angle_center, " Xsec ", xsec_calc)

            # Get TGraph
            gr_name = "beam_" + str(int(beam_bin_center)) + "_energy_" + str(int(energy_center))
            if truth_vars:
                gr_name += "truth_vars"

            xsec_graphs[gr_name] = self.plot_cross_section(xsec, energy_center, angle, beam_bin_center, xsec_xerr,
                                                           xsec_yerr, False)

            true_gr_name = "true_beam_" + str(int(beam_bin_center)) + "_energy_" + str(int(energy_center))

            xsec_graphs[true_gr_name] = self.plot_cross_section(true_xsec, energy_center, angle, beam_bin_center,
                                                                true_xsec_xerr, true_xsec_yerr, True)

        return xsec_graphs

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
                " (T_{#pi^{+}} = " + str(int(beam_energy)) + " [MeV/c])"
        xsec_graph.SetTitle(title)
        xsec_graph.GetXaxis().SetTitle("cos#theta_{#pi^{0}}")
        xsec_graph.GetYaxis().SetTitle("#frac{d^{2}#sigma}{dT_{#pi^{0}}d#Omega_{#pi^{0}}} [#mub/MeV/sr]")

        return xsec_graph

    def get_geant_cross_section(self, energy, angle, beam):
        """
        1) Select the correct 2D plot for the incident pion energy
        2) find the global bin number corresponding to the values of energy, angle
        """
        if int(beam) in self.geant_xsec_dict.keys():
            global_bin = self.geant_xsec_dict[int(beam)].FindBin(energy, angle)
        else:
            print("No match for beam", int(beam), "in dictionary", self.geant_xsec_dict.keys())
            raise RuntimeError
        """
        # 3) get the content in that bin.bin content = coss section[mb]
        """
        return self.geant_xsec_dict[beam].GetBinContent(global_bin) * 1.e3  # convert to micro-barn

    def configure(self, config_file):
        """
        Implement the configuration for the concrete cut class here.
        """

        # with open(config_file, "r") as cfg:
        #     tmp_config = json.load(cfg)
        #     local_config = tmp_config

        beam_bins = np.array([1000, 1400., 1800., 2200.])
        beam_energy_hist = TH1D("beam_energy", "Beam Pi+ Kinetic Energy;T_{#pi^{+}} [MeV/c];Count", len(beam_bins)-1, beam_bins)

        geant_file = "/Users/jsen/tmp_fit/cross_section_cex_n100k_Textended.root"
        geant_xsec_file = TFile(geant_file)

        gROOT.cd()
        for bin_i in range(1, beam_energy_hist.GetXaxis().GetNbins()+1):
            bin_center = beam_energy_hist.GetBinCenter(bin_i)
            geant_graph_name = "inel_cex_" + str(int(bin_center)) + "_MeV"
            print("Loading GEANT Xsec TGraph", geant_graph_name)
            self.geant_xsec_dict[bin_center] = geant_xsec_file.Get(geant_graph_name).Clone()

        geant_xsec_file.Close()

        for k in self.geant_xsec_dict:
            print("Loaded XSec  [ Beam KE=", k, "  Type=", type(self.geant_xsec_dict[k]), "]")

        #return local_config
