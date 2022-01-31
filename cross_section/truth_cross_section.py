import numpy as np
import awkward as ak
import ROOT


class TruthCrossSection:
    """
    It is assumed the events passed to this class are pre-selected to be only sCEX
    """
    def __init__(self, config):
        self.config = config

        self.beam_ke_bins = np.array([1000, 1500., 1800., 2100.])  # MeV/c

        # Compile the c++ code
        self.compile_cpp_helpers()

    @staticmethod
    def compile_cpp_helpers():

        # Fill a histogram with strings
        ROOT.gInterpreter.ProcessLine("""
            void fill_hist_th3d( int len, double *T_pi0, double *theta_pi0, double *T_piplus, TH3D *hist ) {
                for( int i = 0; i < len; i++ ) hist->Fill( T_pi0[i], theta_pi0[i], T_piplus[i] );
           }
        """)

        # Fill beam pion incident histogram
        ROOT.gInterpreter.ProcessLine("""
            void fill_inc_hist( int arr_len, double *emax_arr, double *emin_arr, TH1D *hist ) {
                for( int b = 1; b < hist->GetNbinsX()+1; b++ ) {
                    int bin_center = hist->GetBinCenter(b);
                    double bin_low = hist->GetBinLowEdge(b);
                    double bin_high = hist->GetBinLowEdge(b) + hist->GetBinWidth(b);
                    for( int i = 0; i < arr_len; i++ ) {
                        if (((emin_arr[i] >= bin_low) && (emax_arr[i] <= bin_high)) || ((emin_arr[i] >= bin_low) and (emin_arr[i] <= bin_high))  || 
                            ((emin_arr[i] <= bin_low) && (emax_arr[i] >= bin_high)) || ((emax_arr[i] >= bin_low) && (emax_arr[i] < bin_high))) {
                            hist->Fill(bin_center);
                        }
                    }
                }
            }
        """)

    def extract_truth_xsec(self, events):
        """
        Extract the cross section from the true variables and return a 3D histogram
        :param events:
        :return: TH3D 3D histogram of the cross section variables
        """
        beam_ke = ak.to_numpy(events["true_beam_traj_interacting_Energy"]) #self.beam_end_ke(events)
        pi0_ke = self.pi0_start_ke(events)
        pi0_angle = self.pi0_angle(events)

        beam_ke = self.flatten_and_convert_array(beam_ke)
        pi0_ke = self.flatten_and_convert_array(pi0_ke)
        pi0_angle = self.flatten_and_convert_array(pi0_angle)

        # Plot these variables so we have record of what went into the cross section
        self.plot_xsec_variables(pi0_ke=pi0_ke, pi0_angle=pi0_angle, beam_end_ke=beam_ke)

        return self.bin_cross_section_variables(beam_ke=beam_ke, pi0_ke=pi0_ke, pi0_angle=pi0_angle)

    def plot_xsec_variables(self, pi0_ke, pi0_angle, beam_end_ke):
        print("SHAPES: pi0_ke", pi0_ke.shape, " pi0_angle", pi0_angle.shape, " beam_ke", beam_end_ke.shape)
        # Fill a finely binned histogram so we know the distributions
        beam_ke_fine = ROOT.TH1D("cex_beam_ke_fine", ";Beam KE [MeV/c];Count", 140, 800., 2200.)
        pi0_ke_fine = ROOT.TH1D("cex_pi0_ke_fine", ";#pi0 KE [MeV/c];Count", 100, 0., 1800.)
        pi0_angle_fine = ROOT.TH1D("cex_pi0_angle_fine", ";#pi0 cos#theta;Count", 40, -1., 1.)

        beam_ke_fine.FillN(len(beam_end_ke), beam_end_ke, ROOT.nullptr)
        pi0_ke_fine.FillN(len(pi0_ke), pi0_ke, ROOT.nullptr)
        pi0_angle_fine.FillN(len(pi0_angle), pi0_angle, ROOT.nullptr)

        beam_ke_fine.Write()
        pi0_ke_fine.Write()
        pi0_angle_fine.Write()

    def beam_end_ke(self, events):
        """
        Return beam end KE in MeV/c
        :param events:
        :return:
        """
        beam_end_momentum = "true_beam_endP"
        pi_mass = 0.13957039  # pi+/- [GeV/c]

        events_beam_end_momentum = ak.to_numpy(events[beam_end_momentum])
        return 1000. * (np.sqrt(np.square(pi_mass) + np.square(events_beam_end_momentum)) - pi_mass)

    def pi0_start_ke(self, events):
        """
        Return beam end KE in MeV/c
        :param events:
        :return:
        """
        daughter_start_momentum = "true_beam_daughter_startP"
        daughter_pdg = "true_beam_daughter_PDG"
        pi0_mass = 0.1349768  # pi0 [GeV/c]

        pi0_daughter_mask = events[daughter_pdg] == 111

        events_daughter_start_momentum = ak.to_numpy(events[daughter_start_momentum, pi0_daughter_mask])
        return 1000. * (np.sqrt(np.square(pi0_mass) + np.square(events_daughter_start_momentum)) - pi0_mass)

    def pi0_angle(self, events):

        # Beam direction variables
        beam_end_px = "true_beam_endPx"
        beam_end_py = "true_beam_endPy"
        beam_end_pz = "true_beam_endPz"

        # Daughter pi0 direction variables
        daughter_pdg = "true_beam_daughter_PDG"
        daughter_start_px = "true_beam_daughter_startPx"
        daughter_start_py = "true_beam_daughter_startPy"
        daughter_start_pz = "true_beam_daughter_startPz"

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector and normalize
        beam_dir = np.vstack((ak.to_numpy(events[beam_end_px]),
                              ak.to_numpy(events[beam_end_py]),
                              ak.to_numpy(events[beam_end_pz]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)
        beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)

        # Select only the pi0 daughter
        pi0_daughter_mask = events[daughter_pdg] == 111
        pi0_dir_px = ak.to_numpy(events[daughter_start_px, pi0_daughter_mask])[:,0]
        pi0_dir_py = ak.to_numpy(events[daughter_start_py, pi0_daughter_mask])[:,0]
        pi0_dir_pz = ak.to_numpy(events[daughter_start_pz, pi0_daughter_mask])[:,0]

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        pi0_dir = np.vstack((pi0_dir_px, pi0_dir_py, pi0_dir_pz)).T
        pi0_norm = np.linalg.norm(pi0_dir, axis=1)
        pi0_dir_unit = pi0_dir / np.stack((pi0_norm, pi0_norm, pi0_norm), axis=1)

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        return np.diag(beam_dir_unit @ pi0_dir_unit.T)

    def get_beam_flux(self, events):

        incident_hist = ROOT.TH1D("", "", len(self.beam_ke_bins) - 1, self.beam_ke_bins)
        beam_pion_mask = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic")

        # To get the correct normalization for our cross section the beam incident energy
        # must be entered into the interacting energy been AND all preceding energy bins.
        incident_energy = events["true_beam_incidentEnergies", beam_pion_mask]
        interacting_energy = events["true_beam_interactingEnergy", beam_pion_mask]

        # beam_ke_bins is an array of the bin edges in ascending energy
        for bin, energy in enumerate(self.beam_ke_bins[:-1]):
            beam_incident_in_bin = (interacting_energy <= self.beam_ke_bins[bin+1]) & \
                                   (ak.max(incident_energy) > self.beam_ke_bins[bin])
            incident_hist.SetBinContent(bin, ak.count_nonzero(beam_incident_in_bin))
            print("Bin", bin, ": Bin Edges [", self.beam_ke_bins[bin], ",", self.beam_ke_bins[bin+1],
                  "] Incident pi+ Count [", ak.count_nonzero(beam_incident_in_bin), "]")

        return incident_hist

    def extract_cross_section_slice(self, events):

        avogadro_constant = 6.02214076e23  # 1 / mol
        argon_molar_mass = 39.95           # g / mol
        liquid_argon_density = 1.39        # g / cm^3
        fiducial_thickness = 0.479         # cm

        sigma_factor = argon_molar_mass / (avogadro_constant * liquid_argon_density * fiducial_thickness)
        sigma_factor *= 1.e27 # Convert to milli-barn

        # Select only events with beam pions
        pion_mask = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic")
                    #& (events["true_beam_endZ"] < 222.) & (events["true_beam_endZ"] > 0.)

        # Select only events with >0 incident energy arrays i.e. the event has a beam hit on at least 1 wire
        empty_event_mask = ak.count(events["true_beam_traj_incidentEnergies"], axis=1) > 0

        # Select single charge exchange events, *only* for the interacting events
        cex_mask = events["single_charge_exchange"] == 1

        # Get the max incident energy which is really the pion initial KE
        incident_array = events["true_beam_traj_incidentEnergies", (pion_mask & empty_event_mask)]
        #incident_array = events["true_beam_traj_incidentEnergies", (pion_mask)]

        # Save the start beam KE i.e. initial KE
        initial_ke_hist = ROOT.TH1D("initial_ke_hist", "Initial Beam #pi+ KE;Beam KE [MeV]; Count", 140, 800., 2200.)
        initial_energy_array = ak.to_numpy(events["true_beam_traj_incidentEnergies", (pion_mask & empty_event_mask)][:, 0])
        #initial_energy_array = ak.to_numpy(events["true_beam_traj_incidentEnergies", (pion_mask)][:, 0])
        initial_ke_hist.FillN(len(initial_energy_array), initial_energy_array, ROOT.nullptr)
        initial_ke_hist.Write("initial_ke")

        # Define the incident and interacting histograms
        print("Fine Bins/Lower/Upper ", int((self.beam_ke_bins[-1] - self.beam_ke_bins[0])/50), "/", self.beam_ke_bins[0], "/", self.beam_ke_bins[-1])
        incident_hist = ROOT.TH1D("incident_hist", "Incident Beam #pi+;Beam KE [MeV]; Count",
                                  int((self.beam_ke_bins[-1] - self.beam_ke_bins[0])/50),
                                  self.beam_ke_bins[0], self.beam_ke_bins[-1])
        interacting_hist = ROOT.TH1D("interacting_hist", "Interacting Beam #pi+;Beam KE [MeV]; Count",
                                     int((self.beam_ke_bins[-1] - self.beam_ke_bins[0]) / 50),
                                     self.beam_ke_bins[0], self.beam_ke_bins[-1])

        incident_coarse_hist = ROOT.TH1D("incident_coarse_hist", "Incident Beam #pi+;Beam KE [MeV]; Count",
                                         len(self.beam_ke_bins)-1, self.beam_ke_bins)
        interacting_coarse_hist = ROOT.TH1D("interacting_coarse_hist", "Interacting Beam #pi+;Beam KE [MeV]; Count",
                                            len(self.beam_ke_bins)-1, self.beam_ke_bins)

        #Make sure to use these errors
        interacting_hist.Sumw2(True)
        interacting_coarse_hist.Sumw2(True)

        # Here we create the beam pion incident histogram. Each pion contributes to its interacting
        # bin *and* every preceding energy bin up to its maximum energy (initial energy)
        for b in range(1, incident_hist.GetNbinsX() + 1):
            bin_low = incident_hist.GetBinLowEdge(b)
            bin_high = incident_hist.GetBinLowEdge(b) + incident_hist.GetBinWidth(b)
            bin_mask = (incident_array >= bin_low) & (incident_array < bin_high)
            incident_hist.SetBinContent(b, ak.count_nonzero(bin_mask))

        interacting_array = ak.to_numpy(events["true_beam_traj_interacting_Energy", (cex_mask & pion_mask & empty_event_mask)])
        #interacting_array = ak.to_numpy(events["true_beam_interactingEnergy", (cex_mask & pion_mask & empty_event_mask)])
        #interacting_array = ak.to_numpy(events["true_beam_interactingEnergy", (cex_mask & pion_mask)])

        # Now fill the interacting histogram with the interacting CEX events interaction energy
        interacting_hist.FillN(len(interacting_array), interacting_array, ROOT.nullptr)

        # Save them to file
        incident_hist.Write("incident_hist")
        interacting_hist.Write("interacting_hist")

        # Now we calculate the cross section! F * (N_int / N_inc)  (F = sigma_factor)
        interacting_hist.Divide(incident_hist)
        interacting_hist.Scale(sigma_factor)

        # Tidy it up a bit and write to file
        ROOT.gStyle.SetEndErrorSize(3)
        interacting_hist.SetLineColor(1)
        interacting_hist.SetMarkerStyle(21)

        interacting_hist.GetYaxis().SetRangeUser(0, 200)
        interacting_hist.GetYaxis().SetTitle("MCTruth #sigma_{CEX} [mb]")
        interacting_hist.Write("total_cex_xsec")

        """
        Now redo for the 3 bin histogram
            - Fill 3D hist with interacting, pi0 KE, pi0 cos\theta
        """

        for b in range(1, incident_coarse_hist.GetNbinsX() + 1):
            bin_low = incident_coarse_hist.GetBinLowEdge(b)
            bin_high = incident_coarse_hist.GetBinLowEdge(b) + incident_coarse_hist.GetBinWidth(b)
            bin_mask = (incident_array >= bin_low) & (incident_array < bin_high)
            incident_coarse_hist.SetBinContent(b, ak.count_nonzero(bin_mask))

        # Now fill the interacting histogram with the interacting CEX events interaction energy
        interacting_coarse_hist.FillN(len(interacting_array), interacting_array, ROOT.nullptr)

        # Also save to file
        incident_coarse_hist.Write("incident_coarse_hist")
        interacting_coarse_hist.Write("interacting_coarse_hist")

        # Also calculate this for 3 binned cross section
        # Now we calculate the cross section! F * (N_int / N_inc)  (F = sigma_factor)
        interacting_coarse_hist.Divide(incident_coarse_hist)
        interacting_coarse_hist.Scale(sigma_factor)

        # Tidy it up a bit and write to file
        ROOT.gStyle.SetEndErrorSize(3)
        interacting_coarse_hist.SetLineColor(1)
        interacting_coarse_hist.SetMarkerStyle(21)

        interacting_coarse_hist.GetYaxis().SetRangeUser(0, 200)
        interacting_coarse_hist.GetYaxis().SetTitle("MCTruth #sigma_{CEX} [mb]")
        interacting_coarse_hist.Write("total_cex_coarse_xsec")

        return events[cex_mask & pion_mask & empty_event_mask], interacting_coarse_hist, incident_coarse_hist
        #return events[cex_mask & pion_mask], interacting_coarse_hist, incident_coarse_hist

    def bin_cross_section_variables(self, beam_ke, pi0_ke, pi0_angle):

        #beam_ke_bins = np.array([950., 1550., 1900., 2050.])
        #beam_ke_bins = np.array([950., 1050., 1150., 1250., 1350., 1450., 1550., 1650., 1750., 1850., 1950., 2050])
        pi0_ke_bins = np.array([0., 300., 500., 700., 1200.])  # MeV/c
        pi0_angle_bins = np.array([-1., 0.5, 0.75, 1.])  # cos(angle)

        xsec_hist = ROOT.TH3D("xsec_binning", "XSec Variables;#pi0 KE [MeV/c];#pi0 Angle [rad];Beam Int KE [MeV/c]",
                              len(pi0_ke_bins)-1, pi0_ke_bins, len(pi0_angle_bins)-1, pi0_angle_bins,
                              len(self.beam_ke_bins)-1, self.beam_ke_bins)
        xsec_hist.Sumw2(True)

        print("Shapes: pi0_ke", pi0_ke.shape, " pi0_angle", pi0_angle.shape, " beam_ke", beam_ke.shape)

        # TODO add check to make sure all 3 arrays are the same length

        # ROOT doesn't have a FillN() method for 3D hist. so make our own loop in C++ to fill it
        # Using a double* in c++ (python equiv. float64) so if array is another type, cast it to float64
        if len(beam_ke) > 0 and len(pi0_ke) > 0 and len(pi0_angle) > 0:
            ROOT.fill_hist_th3d(len(pi0_ke), pi0_ke, pi0_angle, beam_ke, xsec_hist)

        return xsec_hist

    @staticmethod
    def flatten_and_convert_array(array):
        """
        Flatten array and convert it to float64 if not already.
        Necessary to work with ROOT FillN method.
        :param array:
        :return:
        """
        array = array.flatten()
        if isinstance(array, np.ndarray):
            if array.dtype != np.float64:
                array.astype('float64')
        else:
            print("Unknown array type!")
        return array
