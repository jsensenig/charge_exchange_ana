import numpy as np
import awkward as ak


class ShowerLikelihood:

    def __init__(self):
        print("Initialized ShowerLikelihood class object!")

        self.no_pi0_template = np.array([])
        self.one_pi0_template = np.array([])
        self.n_pi0_template = np.array([])
        # Load the 0,1,2 pi0 template histograms
        self.load_templates()

        self.Rcut = 150.
        self.pi0_count = np.array([0, 1, 2])

    def classify_npi0(self, r_spacepoints):

        r_mask = r_spacepoints < self.Rcut
        if len(r_spacepoints[r_mask]) < 1:
            return self.pi0_count[0]

        max_r = np.max(ak.to_numpy(r_spacepoints[r_mask]))
        event_hist, _ = np.histogram(ak.to_numpy(r_spacepoints[r_mask]) / max_r, range=[0, 1], bins=50, density=True)

        no_pi0 = self.log_likelihood_shape_test(event_hist, self.no_pi0_template)
        one_pi0 = self.log_likelihood_shape_test(event_hist, self.one_pi0_template)
        n_pi0 = self.log_likelihood_shape_test(event_hist, self.n_pi0_template)
        likelihood_result = np.array([no_pi0, one_pi0, n_pi0])

        return self.pi0_count[likelihood_result == np.min(likelihood_result)]

    def log_likelihood_shape_test(self, hist1, hist2):
        # Integral of each histogram
        integral_h1 = np.sum(hist1)
        integral_h2 = np.sum(hist2)
        # Get the LLR, each list element corresponds to a bin
        llr_list = [self.log_likelihood_ratio(bin_h1, bin_h2, integral_h1, integral_h2) for bin_h1, bin_h2 in zip(hist1, hist2)]
        # Sum the comparison for all k bins
        return -2. * sum(llr_list)

    @staticmethod
    def log_likelihood_ratio(b1, b2, N1, N2):
        # Formulae taken from pg 32 of: http://www.hep.caltech.edu/~fcp/statistics/lectures0802/L0802C.pdf
        if b1 == 0. and b2 == 0.:
            return 0.
        elif b1 == 0.:
            return b2 * np.log(N2 / (N1 + N2))
        elif b2 == 0.:
            return b1 * np.log(N1 / (N1 + N2))
        else:
            term1 = (b1 + b2) * np.log((1 + (b2 / b1)) / (1 + (N2 / N1)))
            term2 = b2 * np.log((N2 * b1) / (N1 * b2))
            return term1 + term2

    def load_templates(self):
        """
        The templates to predict the number of pi0 in an event. Extracted from a full MC sample of 9500 events.
        :return:
        """
        # These 3 arrays are using reco_beam_calo_end{X,Y,Z} which is SCE-corrected while the SPs are not SCE-corrected
        # self.no_pi0_template = np.array([3.07312894e-03, 2.15119026e-02, 5.87735911e-02, 1.43284637e-01,
        #                                  2.30484671e-01, 2.78886452e-01, 3.56098816e-01, 3.91439799e-01,
        #                                  4.28893558e-01, 4.42338497e-01, 5.26849543e-01, 5.48169376e-01,
        #                                  5.75059254e-01, 6.05022261e-01, 6.89533307e-01, 7.18151820e-01,
        #                                  7.77309552e-01, 7.93827621e-01, 7.97476961e-01, 8.27247898e-01,
        #                                  8.60476105e-01, 9.09069956e-01, 8.55482270e-01, 8.88902547e-01,
        #                                  8.85061136e-01, 9.37496399e-01, 9.31158070e-01, 9.56703455e-01,
        #                                  9.62081430e-01, 1.01067528e+00, 1.08750351e+00, 1.08154932e+00,
        #                                  1.04390349e+00, 1.11823479e+00, 1.11151233e+00, 1.10152466e+00,
        #                                  1.20697139e+00, 1.18084980e+00, 1.25287626e+00, 1.29148244e+00,
        #                                  1.34314942e+00, 1.44187369e+00, 1.51409222e+00, 1.51697328e+00,
        #                                  1.60378917e+00, 1.82524652e+00, 1.97717433e+00, 2.29639560e+00,
        #                                  2.81556232e+00, 3.77879618e+00])
        #
        # self.one_pi0_template = np.array([0.00348588, 0.00435735, 0.03340634, 0.07189626, 0.13522307, 0.22687264,
        #                                   0.29760694, 0.35047611, 0.40378102, 0.45069514, 0.56848881, 0.67103176,
        #                                   0.71591246, 0.84924734, 0.89282083, 0.94874015, 1.01119548, 1.0784439,
        #                                   1.18040587, 1.22644853, 1.26363124, 1.313886,   1.37619609, 1.39391598,
        #                                   1.38026295, 1.42093154, 1.37285546, 1.35223067, 1.38723471, 1.36268831,
        #                                   1.35949292, 1.34322548, 1.33901338, 1.37285546, 1.26551943, 1.26566467,
        #                                   1.21846005, 1.2143932,  1.16152403, 1.09674477, 1.15121163, 1.17546754,
        #                                   1.14511134, 1.14670904, 1.17517705, 1.18766812, 1.26203355, 1.27597706,
        #                                   1.39885431, 1.70052811])
        #
        # self.n_pi0_template = np.array([0.00273051, 0.04751087, 0.05242578, 0.11959632, 0.16819939, 0.2544835,
        #                                 0.35933507, 0.44944188, 0.57395312, 0.78147185, 0.93001158, 1.06653706,
        #                                 1.04796959, 1.11022521, 1.13916861, 1.31938225, 1.41713449, 1.35433277,
        #                                 1.32975819, 1.35870159, 1.43679416, 1.4744752,  1.45208502, 1.36689312,
        #                                 1.37071583, 1.3914677,  1.3625243,  1.29098495, 1.37945346, 1.29535376,
        #                                 1.20906966, 1.17958016, 1.07145198, 1.02448721, 1.1621049,  1.01520348,
        #                                 1.0217567,  1.03813976, 1.04742349, 0.95950108, 0.98626008, 0.96769261,
        #                                 0.93929531, 1.02557941, 0.91144411, 0.98626008, 1.03595535, 1.07145198,
        #                                 1.18340287, 1.46082265])

        # These 3 arrays are using reco_beam_end{X,Y,Z}, the UNcorrected beam vertex
        self.no_pi0_template = np.array([0.15040212, 0.36822588, 0.45880056, 0.51899846, 0.53363117, 0.54548553,
                                         0.56715677, 0.63587498, 0.64791456, 0.67884702, 0.68144016, 0.70885335,
                                         0.75645599, 0.78220217, 0.76960692, 0.78590665, 0.85295785, 0.89444809,
                                         0.87073938, 0.9181568,  0.91222962, 0.95853569, 0.91871247, 0.89611511,
                                         0.91482276, 0.91834202, 0.93871669, 0.98076261, 0.99780324, 1.00150773,
                                         0.9692787,  0.97446498, 0.97594677, 0.9876159,  1.02892092, 1.08856314,
                                         1.09726868, 1.13542489, 1.11227185, 1.12912726, 1.15931882, 1.19302964,
                                         1.23489033, 1.29508822, 1.3856629,  1.48790671, 1.66887084, 1.9568946,
                                         2.30715373, 3.24864879])

        self.one_pi0_template = np.array([0.0900085,  0.27253347, 0.40155495, 0.49908738, 0.58435858, 0.65611459,
                                          0.71198674, 0.80422455, 0.88280782, 0.94202394, 0.9566538,  0.99775675,
                                          1.05418623, 1.07090608, 1.15408731, 1.16049658, 1.19714091, 1.1801424,
                                          1.19881289, 1.20550083, 1.20647615, 1.22765462, 1.1975589,  1.16593053,
                                          1.21497541, 1.19198562, 1.20717281, 1.1901743,  1.16913517, 1.16202923,
                                          1.12427025, 1.11409901, 1.0218612,  1.02255786, 1.00221538, 0.97811094,
                                          0.97462764, 0.96166976, 0.91959148, 0.89646236, 0.92669741, 0.96501372,
                                          0.94494991, 0.97588162, 1.01308328, 1.05544022, 1.10114113, 1.19240362,
                                          1.29550933, 1.56093687])

        self.n_pi0_template = np.array([0.15105115, 0.46815511, 0.66990151, 0.72214865, 0.83698891, 0.91665287,
                                        0.93320642, 1.04701208, 1.10650141, 1.09563814, 1.18461347, 1.275658,
                                        1.31548999, 1.26065635, 1.28290018, 1.28703857, 1.27203691, 1.21668598,
                                        1.29221155, 1.29841914, 1.27307151, 1.24772389, 1.28755587, 1.20323622,
                                        1.15202367, 1.107536,   1.10701871, 1.08943056, 1.11788197, 1.01700877,
                                        0.97045191, 0.92544695, 0.98286707, 0.88509767, 0.82250455, 0.86957871,
                                        0.82664294, 0.79612233, 0.85457706, 0.77543039, 0.79405314, 0.75629035,
                                        0.73301192, 0.76767092, 0.75784224, 0.80439911, 0.94769078, 0.98855736,
                                        1.16081775, 1.3454933])