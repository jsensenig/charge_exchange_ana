import numpy as np
import awkward as ak
import ROOT


class TruthCrossSection:
    """
    It is assumed the events passed to this class are pre-selected to be only sCEX
    """
    def __init__(self, config):
        self.config = config

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

    def extract_truth_xsec(self, events):
        """
        Extract the cross section from the true variables and return a 3D histogram
        :param events:
        :return: TH3D 3D histogram of the cross section variables
        """
        beam_ke = self.beam_end_ke(events)
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

    def bin_cross_section_variables(self, beam_ke, pi0_ke, pi0_angle):

        beam_ke_bins = np.array([1000, 1500., 1800., 2100.])  # MeV/c
        #beam_ke_bins = np.array([950., 1550., 1900., 2050.])
        #beam_ke_bins = np.array([950., 1050., 1150., 1250., 1350., 1450., 1550., 1650., 1750., 1850., 1950., 2050])
        pi0_ke_bins = np.array([0., 300., 500., 700., 1200.])  # MeV/c
        pi0_angle_bins = np.array([-1., 0.5, 0.75, 1.])  # cos(angle)

        xsec_hist = ROOT.TH3D("xsec_binning", "XSec Variables;#pi0 KE [MeV/c];#pi0 Angle [rad];Beam KE [MeV/c]",
                              len(pi0_ke_bins)-1, pi0_ke_bins, len(pi0_angle_bins)-1, pi0_angle_bins,
                              len(beam_ke_bins)-1, beam_ke_bins)

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
