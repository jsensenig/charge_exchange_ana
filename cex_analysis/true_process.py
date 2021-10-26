import numpy as np


class TrueProcess:

    def __init__(self):
        self.pdg_dict = {"pi-minus": -211, "anti-muon": -13, "positron": -11, "electron": 11, "muon": 13, "gamma": 22,
                         "pi-zero": 111, "pi-plus": 211, "kaon": 321, "neutron": 2112, "proton": 2212}

    def classify_event_process(self, events):
        """
        Classify events by interaction process. Add one column of booleans per process.
        :param events: Record of events
        :return: Same Record of events but with one additional column per process
        """

        # Count the number of daughter particles of each PDG type
        events = self.get_particle_counts(events)
        print("OUTSIDE FUNC", events["pi-zero-count"].type, " -- ", np.sum(events["pi-zero-count"]))

        #pion_inelastic = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic")
        pion_inelastic = (events["reco_beam_true_byHits_PDG"] == 211) & (events["reco_beam_true_byHits_endProcess"] == "pi+Inelastic")

        # Pion elastic
        events["pion_elastic"] = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+elastic")

        # Pion In-elastic
        events["pion_inelastic"] = pion_inelastic

        # Single charge exchange, the singal
        events["single_charge_exchange"] = pion_inelastic & self.single_charge_exchange(events)

        # Double charge exchange
        events["double_charge_exchange"] = pion_inelastic & self.double_charge_exchange(events)

        # Absorption
        events["absorption"] = pion_inelastic & self.absorption(events)

        # Quasi-elastic
        events["quasi_elastic"] = pion_inelastic & self.quasi_elastic(events)

        # Pion production
        events["pion_production"] = pion_inelastic & self.pion_production(events)

        # pi0 production
        events["pi0_production"] = pion_inelastic & self.pi0_and_pion(events)

        # Pion and pi0
        events["pion_and_pi0"] = pion_inelastic & self.pi0_and_pion(events)

        return events

    '''
    Below are the definitions of the true processes
    '''

    @staticmethod
    def get_process_list():
        return ["pion_inelastic", "pion_elastic", "single_charge_exchange", "double_charge_exchange", "absorption",
                "quasi_elastic", "pion_production", "pi0_production", "pion_and_pi0"]

    def get_particle_counts(self, events):
        for pdg in self.pdg_dict:
            if pdg == "pi-zero":
                events[pdg + "-count"] = np.count_nonzero(
                    events["reco_daughter_PFP_true_byHits_PDG"] == self.pdg_dict[pdg], axis=1)
            else:
                tmp = np.count_nonzero(events["reco_daughter_PFP_true_byHits_PDG"] == self.pdg_dict[pdg], axis=1)
                print("COUNTING:", pdg + "-count     --> ", tmp.type, "  -- ", tmp[4])
                events[pdg + "-count"] = np.count_nonzero(events["reco_daughter_PFP_true_byHits_PDG"] == self.pdg_dict[pdg], axis=1)

        return events

    @staticmethod
    def mask_daughter_momentum(events, momentum_threshold, pdg_select):
        momentum_var = "true_beam_daughter_startP"
        daughter_pdg_var = "true_beam_daughter_PDG"
        # Momentum in GeV
        mom_mask = (events[momentum_var] > momentum_threshold) & (events[daughter_pdg_var] == pdg_select)
        return np.count_nonzero(mom_mask, axis=1)

    @staticmethod
    def single_charge_exchange(events):
        selected_pi0 = TrueProcess.mask_daughter_momentum(events=events, momentum_threshold=0.0, pdg_select=111)
        selected_pi_plus = TrueProcess.mask_daughter_momentum(events=events, momentum_threshold=0.0, pdg_select=211)
        selected_pi_minus = TrueProcess.mask_daughter_momentum(events=events, momentum_threshold=0.0, pdg_select=-211)
        return (selected_pi_plus == 0) & (selected_pi_minus == 0) & (selected_pi0 == 1) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    # def single_charge_exchange(self, events):
    #     selected_pi0 = events["pi-zero-count"]
    #     selected_pi_plus = events["pi-plus-count"]
    #     selected_pi_minus = events["pi-minus-count"]
    #     return (selected_pi_plus == 0) & (selected_pi_minus == 0) & (selected_pi0 == 1) & \
    #            ((events["proton-count"] > 0) | (events["neutron-count"] > 0))

    @staticmethod
    def double_charge_exchange(events):
        return (events["true_daughter_nPiMinus"] == 1) & (events["true_daughter_nPiPlus"] == 0) & \
               (events["true_daughter_nPi0"] == 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def absorption(events):
        return (events["true_daughter_nPiMinus"] == 0) & (events["true_daughter_nPiPlus"] == 0) & \
               (events["true_daughter_nPi0"] == 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def quasi_elastic(events):
        return (events["true_daughter_nPiMinus"] == 0) & (events["true_daughter_nPiPlus"] == 1) & \
               (events["true_daughter_nPi0"] == 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def pion_production(events):
        return (((events["true_daughter_nPiMinus"] > 1) | (events["true_daughter_nPiPlus"] > 1)) |
                ((events["true_daughter_nPiMinus"] > 0) & (events["true_daughter_nPiPlus"] > 0))) & \
                (events["true_daughter_nPi0"] == 0) & \
                ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def pi0_production(events):
        return (events["true_daughter_nPiMinus"] == 0) & (events["true_daughter_nPiPlus"] == 0) & \
               (events["true_daughter_nPi0"] > 1) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def pi0_and_pion(events):
        return ((events["true_daughter_nPiMinus"] > 0) | (events["true_daughter_nPiPlus"] > 0)) & \
               (events["true_daughter_nPi0"] > 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

