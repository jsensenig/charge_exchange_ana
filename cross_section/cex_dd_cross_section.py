from cross_section.truth_cross_section import TruthCrossSection
from cross_section.reco_cross_section import RecoCrossSection
from cex_analysis.true_process import TrueProcess
from ROOT import TMath, TGraph, TGraphErrors, TFile, TH1D, TH2D, gROOT, nullptr
import awkward as ak
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

        beam_ke_bins = np.array([1000, 1500., 1800., 2100.])  # MeV/c
        #beam_ke_bins = np.array([950., 1050., 1150., 1250., 1350., 1450., 1550., 1650., 1750., 1850., 1950., 2050])
        self.beam_ke_hist = TH1D("beam_ke", "Beam KE [MeV/c];Count", len(beam_ke_bins)-1, beam_ke_bins)

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
        valid_piplus = TrueProcess.mask_daughter_momentum(events=all_events,
                momentum_threshold=0.0, pdg_select=211)
        valid_piminus = TrueProcess.mask_daughter_momentum(events=all_events,
                momentum_threshold=0.0, pdg_select=-211)
        true_scex_mask = TrueProcess.single_charge_exchange(all_events, valid_piplus, valid_piminus)
        pion_inelastic = (all_events["true_beam_PDG"] == 211) & (all_events["true_beam_endProcess"] == "pi+Inelastic") \
                         & (all_events["true_beam_endZ"] < 222.)
        true_scex_mask = true_scex_mask & pion_inelastic

        print("TRUE NUMBER OF sCEX EVENTS", np.count_nonzero(true_scex_mask))

        #if self.config["run_truth_xsec"]:
        if True:
            truth_3d_hist = self.run_truth_cross_section(events=all_events[true_scex_mask])
            beam_pions = all_events["true_beam_PDG"] == 211
            incident_pions = np.count_nonzero(beam_pions)
            print("NUMBER OF INCIDENT PI+", incident_pions)

            beam_pions = beam_pions & (all_events["true_beam_endProcess"] == "pi+Inelastic")
            incident_pions = np.count_nonzero(beam_pions)
            print("NUMBER OF INCIDENT PI+", incident_pions)

            beam_pions = beam_pions & (all_events["true_beam_endZ"] > 0.)
            incident_pions = np.count_nonzero(beam_pions)
            print("NUMBER OF INCIDENT PI+", incident_pions)

            decay_pions = (all_events["true_beam_PDG"] == 211) & (all_events["true_beam_endProcess"] == "Decay")
            incident_decay_pions = np.count_nonzero(decay_pions)
            print("NUMBER OF INCIDENT PI+ Decays", incident_decay_pions)

            #########
            # Bin the incident beam pions here
            self.bin_beam_interaction_ke(all_events[beam_pions])
            ########

            self.calculate_double_differential_xsec(truth_3d_hist, incident_pions, True)

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
 
    def bin_beam_interaction_ke(self, events):
        """
        Return beam end KE in MeV/c
        :param events:
        :return:
        """
        beam_end_momentum = "true_beam_endP"
        pi_mass = 0.13957039  # pi+/- [GeV/c]
                                                                                                     
        events_beam_end_momentum = ak.to_numpy(events[beam_end_momentum])
        beam_end_ke = 1000. * (np.sqrt(np.square(pi_mass) + np.square(events_beam_end_momentum)) - pi_mass)

        self.beam_ke_hist.FillN(len(beam_end_ke), beam_end_ke, nullptr)
        # Write it to file, so we have record of the flux distribution
        self.beam_ke_hist.Write()

        # Fill a finely binned histogram so we know the distribution of KE
        beam_ke_fine = TH1D("beam_ke_fine", ";Beam KE [MeV/c];Count", 140, 800., 2200.)
        beam_ke_fine.FillN(len(beam_end_ke), beam_end_ke, nullptr)
        beam_ke_fine.Write()

    def calculate_double_differential_xsec(self, xsec_3d_hist, total_events, truth_vars):
        """
        Double differential cross section wrt angle
        X = pi0 KE
        Y = pi0 angle
        Z = beam pi+ interaction KE
        xsec calculation: https://ir.uiowa.edu/cgi/viewcontent.cgi?article=1518&context=etd(pg54)
        """
        xsec_graphs = {}

        avogadro_constant    = 6.02214076e23  # 1 / mol
        argon_molar_mass     = 39.95          # g / mol
        liquid_argon_density = 1.39           # g / cm^3
        fiducial_thickness   = 230.5          # 222 cm

        # N_tgt = Ar atomic mass / (Avagadro's number * LAr density) = 39.9624 / (6.022e23) * (1.3973) = 4.7492e-23
        num_target = argon_molar_mass / (avogadro_constant * liquid_argon_density * fiducial_thickness)

        # Get the number of bins in energy and angle
        energy_bins = xsec_3d_hist.GetNbinsX()
        angle_bins  = xsec_3d_hist.GetNbinsY()
        beam_bins   = xsec_3d_hist.GetNbinsZ()

        for beambin_i in range(1, beam_bins+1):  # beam KE

            beam_bin_center = xsec_3d_hist.GetZaxis().GetBinCenter(beambin_i)

            # We use total events from the give beam KE bin
            beam_flux = self.beam_ke_hist.GetBinContent(beambin_i)
            print("**********************************************")
            print("* \033[92m Beam pi+ KE:", beam_bin_center, " Flux:", beam_flux, "\033[0m")
            print("**********************************************")

            # Cross section plotted wrt to pi0 KE
            self.plot_xsec_energy(xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                                  beam_flux, xsec_graphs, truth_vars)
            # Cross section plotted wrt to pi0 cos(theta)
            self.plot_xsec_angle(xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                                 beam_flux, xsec_graphs, truth_vars)

        self.write_graphs_to_file(xsec_graphs)

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
                gr_name += "_truth_vars"

            xsec_graphs[gr_name] = self.plot_cross_section(xsec, energy, angle_center, beam_bin_center, xsec_xerr,
                                                           xsec_yerr, False, "Energy")

            xsec_graphs["true_" + gr_name] = self.plot_cross_section(true_xsec, energy, angle_center, beam_bin_center,
                                                                     true_xsec_xerr, true_xsec_yerr, True, "Energy")

        #return xsec_graphs

    def plot_xsec_angle(self, xsec_3d_hist, beambin_i, num_target, beam_bin_center, energy_bins, angle_bins,
                        total_events, xsec_graphs, truth_vars):

        for energybin_i in range(1, energy_bins+1):  # energy xsec
            # Get the energy bin center and width
            energy_center = xsec_3d_hist.GetXaxis().GetBinCenter(energybin_i)
            ebin_width = xsec_3d_hist.GetXaxis().GetBinWidth(energybin_i)

            # Project out the angle so we can get the error bars
            h_xerr = xsec_3d_hist.ProjectionY("h_xerr", energybin_i, energybin_i, beambin_i, beambin_i, "e")
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

                angle.append(angle_center)
                xsec_xerr.append(abin_width / 2.)

                print("Ebin width ", ebin_width, " Abin width ", abin_width, " N_int ", num_interactions, " Energy ",
                      energy_center, " Angle ", angle_center, " Xsec ", xsec_calc)

            # Get TGraph
            gr_name = "beam_" + str(int(beam_bin_center)) + "_energy_" + str(int(energy_center))
            if truth_vars:
                gr_name += "_truth_vars"

            # Graph the cross section extracted from the events
            xsec_graphs[gr_name] = self.plot_cross_section(xsec, energy_center, angle, beam_bin_center, xsec_xerr,
                                                           xsec_yerr, False, "Angle")

            # Graph the Geant nominal cross section
            xsec_graphs["true_" + gr_name] = self.plot_cross_section(true_xsec, energy_center, angle, beam_bin_center,
                                                                     true_xsec_xerr, true_xsec_yerr, True, "Angle")

        #return xsec_graphs

    def plot_cross_section(self, xsec, energy, angle, beam_energy, xerr, yerr, true_xsec, xaxis):

        print("Writing Xsec to file")

        if xaxis == "Energy":
            angle_degrees = str(int(TMath.ACos(angle) * TMath.RadToDeg()))
            title = "#pi^{+} CEX Cross-section (#theta_{#pi^{0}} = " + angle_degrees + " [deg])" + \
                    " (T_{#pi^{+}} = " + str(int(beam_energy)) + " [MeV/c])"
            xaxis_title = "T_{#pi^{0}} [MeV]"
            xsec_graph = TGraphErrors(len(energy), np.asarray(energy), np.asarray(xsec), np.asarray(xerr),
                                      np.asarray(yerr))
        elif xaxis == "Angle":
            title = "#pi^{+} CEX Cross-section (T_{#pi^{0}} = " + str(int(energy)) + " [MeV/c])" + \
                    " (T_{#pi^{+}} = " + str(int(beam_energy)) + " [MeV/c])"
            xaxis_title = "cos#theta_{#pi^{0}}"
            xsec_graph = TGraphErrors(len(angle), np.asarray(angle), np.asarray(xsec), np.asarray(xerr),
                                      np.asarray(yerr))
        else:
            print("Unknown X-axis plot", xaxis)
            raise RuntimeError

        xsec_graph.SetLineWidth(1)
        xsec_graph.SetMarkerStyle(8)
        xsec_graph.SetMarkerSize(0.5)
        if true_xsec:
            xsec_graph.SetLineColor(64)
        else:
            xsec_graph.SetLineColor(46)

        xsec_graph.SetTitle(title)
        xsec_graph.GetXaxis().SetTitle(xaxis_title)
        xsec_graph.GetYaxis().SetTitle("#frac{d^{2}#sigma}{dT_{#pi^{0}}d#Omega_{#pi^{0}}} [#mub/MeV/sr]")

        return xsec_graph

    def get_geant_cross_section(self, energy, angle, beam):
        # If the beam KE does not exist we should know, throw error
        if int(beam) not in self.geant_xsec_dict.keys():
            print("No match for beam", int(beam), "in dictionary", self.geant_xsec_dict.keys())
            raise RuntimeError

        """
        1) Select the correct 2D plot for the incident pion energy
        2) find the global bin number corresponding to the values of energy, angle
        """
        global_bin = self.geant_xsec_dict[int(beam)].FindBin(energy, angle)
        """
        # 3) get the content in that bin.bin content = coss section[mb]
        """
        return self.geant_xsec_dict[beam].GetBinContent(global_bin) * 1.e3  # convert to micro-barn

    def configure(self, config_file):
        """
        Implement the configuration for the concrete cut class here.
        """
        beam_bins = np.array([1000, 1500., 1800., 2100.])
        #beam_bins = np.array([950.,1050.,1150.,1250.,1350.,1450.,1550.,1650.,1750.,1850.,1950.,2050])
        beam_energy_hist = TH1D("beam_energy", "Beam Pi+ Kinetic Energy;T_{#pi^{+}} [MeV/c];Count", len(beam_bins)-1, beam_bins)

        #geant_file = "/Users/jsen/tmp_fit/cross_section_cex_n1m_Textended.root"
        #geant_file = "/Users/jsen/tmp_fit/cross_section_out_1m_3ke.root" # 1.2,1.6,2.0 GeV
        geant_file = "/Users/jsen/tmp_fit/cross_section_out_1m_3ke_x50.root" # 1.25,1.65,1.95 GeV
        geant_xsec_file = TFile(geant_file)

        # cd into the ROOT directory so we can clone the histograms from file
        # otherwise we lose the histograms when the file is closed
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
