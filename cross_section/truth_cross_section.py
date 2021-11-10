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

        return self.bin_cross_section_variables(beam_ke=beam_ke, pi0_ke=pi0_ke, pi0_angle=pi0_angle)

    def beam_end_ke(self, events):
        """
        Return beam end KE in MeV/c
        :param events:
        :return:
        """
        beam_end_momentum = "true_beam_endP"
        pi_mass = 0.13957039  # pi+/- [GeV/c]

        events_beam_end_momentum = ak.to_numpy(events[beam_end_momentum])
        return 1000. * np.sqrt(np.square(pi_mass) + np.square(events_beam_end_momentum)) - pi_mass

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

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        # beam_dir = np.vstack(ak.to_numpy([events[beam_end_px], events[beam_end_py], events[beam_end_pz]]).T)
        beam_dir = np.vstack((ak.to_numpy(events[beam_end_px]), ak.to_numpy(events[beam_end_py]), ak.to_numpy(events[beam_end_pz]))).T
        beam_norm = np.linalg.norm(beam_dir, axis=1)
        beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)

        # Select only the pi0 daughter
        pi0_daughter_mask = events[daughter_pdg] == 111
        pi0_dir_px = ak.to_numpy(events[daughter_start_px, pi0_daughter_mask])
        pi0_dir_py = ak.to_numpy(events[daughter_start_py, pi0_daughter_mask])
        pi0_dir_pz = ak.to_numpy(events[daughter_start_pz, pi0_daughter_mask])

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        pi0_dir = np.vstack([pi0_dir_px, pi0_dir_py, pi0_dir_pz])
        pi0_norm = np.linalg.norm(pi0_dir, axis=1)
        print(pi0_dir.shape, " ", np.stack((pi0_norm, pi0_norm, pi0_norm), axis=1).shape)
        pi0_dir_unit = pi0_dir / np.stack((pi0_norm, pi0_norm, pi0_norm), axis=1)

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        return np.diag(beam_dir_unit @ pi0_dir_unit.T)

    def bin_cross_section_variables(self, beam_ke, pi0_ke, pi0_angle):

        beam_ke_bins = np.array([1000., 1400., 1800., 2200.])  # MeV/c
        pi0_ke_bins = np.array([0., 300., 500., 700., 1200.])  # MeV/c
        pi0_angle_bins = np.array([-1., 0.5, 0.75, 1.])  # cos(angle)

        xsec_hist = ROOT.TH3D("xsec_binning", "XSec Variables;#pi0 KE [MeV/c];#pi0 Angle [rad];Beam KE [MeV/c]",
                              len(pi0_ke_bins)-1, pi0_ke_bins, len(pi0_angle_bins)-1, pi0_angle_bins,
                              len(beam_ke_bins)-1, beam_ke_bins)

        pi0_ke = pi0_ke.flatten()
        pi0_angle = pi0_angle.flatten()
        beam_ke = beam_ke.flatten()

        # TODO add check to make sure all 3 arrays are the same length

        # ROOT doesn't have a FillN() method for 3D hist. so make our own loop in C++ to fill it
        # Using a double* in c++ (python equiv. float64) so if array is another type, cast it to float64
        if len(beam_ke) > 0 and len(pi0_ke) > 0 and len(pi0_angle) > 0:
            if isinstance(beam_ke, np.ndarray):
                if beam_ke.dtype != np.float64:
                    pi0_ke.astype('float64')
                    pi0_angle.astype('float64')
                    beam_ke.astype('float64')
                ROOT.fill_hist_th3d(len(pi0_ke), pi0_ke, pi0_angle, beam_ke, xsec_hist)
            else:
                print("Unknown array type!")

        return xsec_hist
