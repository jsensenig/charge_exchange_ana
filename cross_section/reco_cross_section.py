import numpy as np
import awkward as ak
from ROOT import TH3D, nullptr


class RecoCrossSection:

    def __init__(self, config):
        self.config = config
        self.pi_mass = 0.13957039  # pi+/- [GeV/c]
        self.pi0_mass = 0.1349768  # pi0 [GeV/c]

    def extract_reco_xsec(self, events):
        """
        Extract the cross section from the reco variables and return a 3D histogram
        :param events:
        :return: TH3D 3D histogram of the cross section variables
        """
        beam_ke = self.beam_end_ke(events)
        pi0_ke = self.pi0_start_ke(events)
        pi0_angle = self.pi0_angle(events)
        return self.bin_cross_section_variables(beam_ke=beam_ke, pi0_ke=pi0_ke, pi0_angle=pi0_angle)

    def beam_end_ke(self, events):
        """
        Return beam end KE in MeV/c and defined as
        End KE = E_initial - Sum(Energy Loss in TPC)
        :param events:
        :return:
        """
        beam_end_momentum = "true_beam_endP"

        # Get the initial beam energy from the beam instrumentation
        beam_instr_momentum = "beam_inst_P"
        beam_dedx_tpc = "reco_beam_calibrated_dEdX_SCE"

        pion_ke = 1000. * (np.sqrt(np.square(self.pi_mass, 2) + np.square((events[beam_instr_momentum] + 1.), 2)) - self.pi_mass)
        pion_ke -= np.sum(events[beam_dedx_tpc, events[beam_dedx_tpc] < 1000.], axis=1)

        return pion_ke

    def find_pi0_gammas(self, events):
        selection_mask = self.shower_selection(events=events)

    def shower_selection(self, events):
        """
        1. CNN cut to select shower-like daughters
        2. Energy cut, eliminate small showers from e.g. de-excitation gammas
        :param events:
        :return:
        """
        # 2 step cut (track score and min shower energy) on daughter showers, get a mask from each
        cnn_shower_mask = events["reco_daughter_PFP_trackScore_collection"] < 0.5
        min_shower_energy = events["reco_daughter_allShower_energy"] > 10.0

        # Return shower selection mask
        return cnn_shower_mask & min_shower_energy

    def pi0_start_ke(self, events):
        """
        Return beam end KE in MeV/c
        :param events:
        :return:
        """
        #return events[self.config["beam_momentum_var"]]
        daughter_start_momentum = "true_beam_daughter_startP"
        daughter_pdg = "true_beam_daughter_PDG"

        pi0_daughter_mask = events[daughter_pdg] == 111

        return (np.sqrt(np.square(self.pi0_mass, 2) + np.square(events[daughter_start_momentum, pi0_daughter_mask], 2))
                - self.pi0_mass) * 1000.

    def pi0_angle(self, events):

        # Beam direction variables
        beam_end_px = "beam_end_dirX"
        beam_end_py = "beam_end_dirY"
        beam_end_pz = "beam_end_dirZ"

        # Daughter pi0 direction variables
        daughter_pdg = "true_beam_daughter_PDG"
        daughter_start_px = "true_beam_daughter_startPx"
        daughter_start_py = "true_beam_daughter_startPy"
        daughter_start_pz = "true_beam_daughter_startPz"

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        beam_dir = np.vstack(ak.to_numpy([events[beam_end_px], events[beam_end_py], events[beam_end_pz]]).T)
        beam_norm = np.linalg.norm(beam_dir, axis=1)
        beam_dir_unit = beam_dir / np.stack((beam_norm, beam_norm, beam_norm), axis=1)

        # Select only the pi0 daughter
        pi0_daughter_mask = events[daughter_pdg] == 111
        pi0_dir_px = events[daughter_start_px, pi0_daughter_mask]
        pi0_dir_py = events[daughter_start_py], pi0_daughter_mask
        pi0_dir_pz = events[daughter_start_pz, pi0_daughter_mask]

        # Convert to numpy array and combine from (N,1) to (N,3) shape, i.e. each row is a 3D vector
        # and normalize
        pi0_dir = np.vstack(ak.to_numpy([pi0_dir_px, pi0_dir_py, pi0_dir_pz]).T)
        pi0_norm = np.linalg.norm(pi0_dir, axis=1)
        pi0_dir_unit = pi0_dir / np.stack((pi0_norm, pi0_norm, pi0_norm), axis=1)

        # Calculate the cos angle between beam and pi0 direction by taking the dot product of their
        # respective direction unit vectors
        return np.diag(beam_dir_unit @ pi0_dir_unit.T)

    def bin_cross_section_variables(self, beam_ke, pi0_ke, pi0_angle):

        beam_ke_bins = [1000., 1400., 2200.]  # MeV/c
        pi0_ke_bins = [0., 300., 500., 700., 1200.]  # MeV/c
        pi0_angle_bins = [-1., 0.5, 0.75, 1.]  # cos(angle)

        xsec_hist = TH3D("xsec_binning", "XSec Variables;#pi0 KE [MeV/c];#pi0 Angle [rad];Beam KE [MeV/c]",
                         pi0_ke_bins, pi0_angle_bins, beam_ke_bins)

        pi0_ke_flat = ak.flatten(pi0_ke, axis=None)
        pi0_angle_flat = ak.flatten(pi0_angle, axis=None)
        beam_ke_flat = ak.flatten(beam_ke, axis=None)

        # Just a loop in c++ which does hist.FillN() (nullptr sets weights = 1)
        # FillN() only likes double* (python float64) so if array is another type, cast it to float64
        if len(beam_ke_flat) > 0 and len(pi0_ke_flat) > 0 and len(pi0_angle_flat) > 0:
            if isinstance(beam_ke_flat, ak.Array):
                if ak.type(beam_ke_flat.layout).dtype != 'float64':
                    pi0_ke_flat = ak.values_astype(pi0_ke_flat, np.float64)
                    pi0_angle_flat = ak.values_astype(pi0_angle_flat, np.float64)
                    beam_ke_flat = ak.values_astype(beam_ke_flat, np.float64)
                xsec_hist.FillN(len(beam_ke_flat), ak.to_numpy(pi0_ke_flat), ak.to_numpy(pi0_angle_flat),
                                ak.to_numpy(beam_ke_flat), nullptr)
            elif isinstance(beam_ke_flat, np.ndarray):
                if beam_ke_flat.dtype != np.float64:
                    pi0_ke_flat = ak.values_astype(pi0_ke_flat, np.float64)
                    pi0_angle_flat = ak.values_astype(pi0_angle_flat, np.float64)
                    beam_ke_flat = ak.values_astype(beam_ke_flat, np.float64)
                xsec_hist.FillN(len(beam_ke_flat), ak.to_numpy(pi0_ke_flat), ak.to_numpy(pi0_angle_flat),
                                ak.to_numpy(beam_ke_flat), nullptr)
            else:
                print("Unknown array type!")

        return xsec_hist

