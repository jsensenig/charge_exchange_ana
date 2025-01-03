import awkward as ak
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
        events = self.get_reco_particle_counts(events)
        events["true_daughter_nKaon"] = ak.count_nonzero(events["true_beam_daughter_PDG"] == 321, axis=1)



        beam_matched = events["reco_beam_true_byE_matched"]
        # cosmic_origin = events["reco_beam_true_byE_origin"]
        #events["misid_pion"] = ~beam_matched & (events["true_beam_PDG"] == 211)
        #events["misid_proton"] = ~beam_matched & (events["true_beam_PDG"] == 2212)
        #events["misid_muon"] = ~beam_matched & (np.abs(events["true_beam_PDG"]) == 13)
        #events["misid_electron_gamma"] = ~beam_matched & ((np.abs(events["true_beam_PDG"]) == 11) | (events["true_beam_PDG"]) == 22)
        #events["matched_pion"] = beam_matched & (events["true_beam_PDG"] == 211)
        #events["matched_muon"] = beam_matched & (np.abs(events["true_beam_PDG"]) == 13)
        #events["misid_beam"] = events["misid_pion"] | events["misid_proton"] | events["misid_muon"] | events["misid_electron_gamma"]

        events["misid_pion"] = (np.abs(events["true_beam_PDG"]) == 211) & ~(events["true_beam_endProcess"] == "pi+Inelastic")
        events["misid_proton"] = (events["true_beam_PDG"] == 2212)
        events["matched_proton"] = (events["true_beam_PDG"] == 2212) & (events["true_beam_endProcess"] == "protonInelastic")
        #events["misid_muon"] = (np.abs(events["true_beam_PDG"]) == 13)
        events["misid_electron_gamma"] = ((np.abs(events["true_beam_PDG"]) == 11) | (events["true_beam_PDG"]) == 22)
        events["matched_pion"] = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic")
        events["matched_muon"] = (np.abs(events["true_beam_PDG"]) == 13)
        events["misid_beam"] = events["misid_pion"] | events["misid_proton"] | events["misid_electron_gamma"]


        # Momentum cut on true charged pions, i.e., pions with momentum < cut momentum are indistinguishable from say
        # protons and other charged particles. set to 0.125
        valid_piplus = TrueProcess.mask_daughter_momentum(events=events, momentum_threshold=-1000., pdg_select=211)
        valid_piminus = TrueProcess.mask_daughter_momentum(events=events, momentum_threshold=-1000., pdg_select=-211)

        # Pion inelastic
        pion_inelastic = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic") #& ~events["misid_beam"]
        proton_inelastic = (events["true_beam_PDG"] == 2212) & (events["true_beam_endProcess"] == "protonInelastic") 

        # Pion elastic
        events["pion_elastic"] = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+elastic")

        # Pion In-elastic
        events["pion_inelastic"] = pion_inelastic
        events["proton_inelastic"] = proton_inelastic

        # Single charge exchange, the singal
        events["single_charge_exchange"] = pion_inelastic & self.single_charge_exchange(events, valid_piplus, valid_piminus)
        events["proton_charge_exchange"] = proton_inelastic & self.single_charge_exchange(events, valid_piplus, valid_piminus)

        # Double charge exchange
        events["double_charge_exchange"] = pion_inelastic & self.double_charge_exchange(events, valid_piplus, valid_piminus)

        # Absorption
        events["absorption"] = pion_inelastic & self.absorption(events, valid_piplus, valid_piminus)

        # Quasi-elastic
        events["quasi_elastic"] = pion_inelastic & self.quasi_elastic(events, valid_piplus, valid_piminus)

        # Pion production
        events["pion_production"] = pion_inelastic & self.pion_production(events, valid_piplus, valid_piminus)
        events["all_pion_production"] = pion_inelastic & self.pion_production_v2(events, valid_piplus, valid_piminus)

        # pi0 production
        events["pi0_production"] = pion_inelastic & self.pi0_production(events, valid_piplus, valid_piminus)

        # Pion and pi0
        events["pion_and_pi0"] = pion_inelastic & self.pi0_and_pion(events, valid_piplus, valid_piminus)

        # Combining all multiple pions together
        events["charged_neutral_pion_production"] = pion_inelastic & self.charged_neutral_pion_production(events, valid_piplus, valid_piminus)

        # 1 charged and neutral pion, charged pion NOT reconstructed (only in truth)
        events["mctruth_charged_neutral_pion"] = pion_inelastic & self.mctruth_charged_neutral_pion(events, valid_piplus, valid_piminus)

        # 1 charged and neutral pion, charged pion IS reconstructed (in truth AND reco)
        events["mcreco_charged_neutral_pion"] = pion_inelastic & self.mcreco_charged_neutral_pion(events)

        ## Daughter backgrounds
        events["daughter_zero_pi0_bkgd"] = pion_inelastic & self.daughter_charged_pion_prod(events, valid_piplus, valid_piminus)
        events["daughter_one_pi0_bkgd"] = pion_inelastic & self.daughter_one_pi0_bkgd(events, valid_piplus, valid_piminus)
        events["daughter_n_pi0_bkgd"] = pion_inelastic & self.daughter_n_pi0_bkgd(events, valid_piplus, valid_piminus)
        events["beam_bkgd"] = ~pion_inelastic 
        events["daughter_other_bkgd"] =  pion_inelastic & (~events["daughter_zero_pi0_bkgd"] & ~events["daughter_one_pi0_bkgd"] &
                                          ~events["daughter_n_pi0_bkgd"] & ~events["single_charge_exchange"])

        events["proton_zero_pi0_bkgd"] = proton_inelastic & self.daughter_charged_pion_prod(events, valid_piplus, valid_piminus)
        events["proton_one_pi0_bkgd"] = proton_inelastic & self.daughter_one_pi0_bkgd(events, valid_piplus, valid_piminus)
        events["proton_n_pi0_bkgd"] = proton_inelastic & self.daughter_n_pi0_bkgd(events, valid_piplus, valid_piminus)
        events["proton_zero_pion_bkgd"] = proton_inelastic & self.absorption(events, valid_piplus, valid_piminus)
        events["proton_beam_bkgd"] = ~proton_inelastic 
        events["proton_other_bkgd"] =  proton_inelastic & (~events["proton_zero_pi0_bkgd"] & ~events["proton_one_pi0_bkgd"] &
                                          ~events["proton_n_pi0_bkgd"] & events["proton_zero_pion_bkgd"] & ~events["proton_charge_exchange"])

        # other (fill "other" column with all zeroes)
        # events["other"] = np.zeros_like(events["pion_inelastic"])
        # if event not already in a category, classify as "other"
        # events["other"] = ~np.any(events[self.get_process_list_simple()[:-1]])

        proc_mask = np.zeros(len(events)).astype(bool)
        for proc in self.get_process_list_simple():
            if proc == 'simple_other': continue
            proc_mask |= ak.to_numpy(events[proc])

        events["simple_other"] = ~proc_mask

        proc_mask = np.zeros(len(events)).astype(bool)
        for proc in self.get_beam_particle_list():
            if proc == 'beam_other': continue
            proc_mask |= ak.to_numpy(events[proc])

        events["beam_other"] = ~proc_mask

        return events

    '''
    Below are the definitions of the true processes
    '''

    @staticmethod
    def get_process_list():
        # return ["single_charge_exchange", "double_charge_exchange", "absorption",
        #         "quasi_elastic", "pion_production", "pi0_production", "pion_and_pi0"]
        return ["single_charge_exchange", "double_charge_exchange", "absorption", "quasi_elastic", "simple_other",
                "pion_production", "pi0_production", "mctruth_charged_neutral_pion", "mcreco_charged_neutral_pion"]

    @staticmethod
    def get_process_list_simple():
        return ["single_charge_exchange", "double_charge_exchange", "absorption", "quasi_elastic", "pi0_production",
                "daughter_zero_pi0_bkgd", "misid_beam", "simple_other"]

    @staticmethod
    def get_daughter_bkgd_list():
        return ["single_charge_exchange", "daughter_zero_pi0_bkgd", "daughter_one_pi0_bkgd", "daughter_n_pi0_bkgd", "daughter_other_bkgd", "beam_bkgd"]

    @staticmethod
    def get_beam_particle_list():
        return ["misid_pion", "misid_proton", "misid_electron_gamma", "matched_pion",
                "matched_muon", "beam_other"]

    @staticmethod
    def get_proton_bkgd_list():
        return ["proton_charge_exchange", "proton_zero_pi0_bkgd", "proton_one_pi0_bkgd", "proton_n_pi0_bkgd", "proton_zero_pion_bkgd", "proton_beam_bkgd", "proton_other_bkgd"]

    def get_reco_particle_counts(self, events):
            for pdg in self.pdg_dict:
                if pdg == "pi-zero":
                    events["mcreco-" + pdg + "-count"] = np.count_nonzero(events["reco_daughter_PFP_true_byHits_PDG"] == self.pdg_dict[pdg], axis=1)
                else:
                    events["mcreco-" + pdg + "-count"] = np.count_nonzero(events["reco_daughter_PFP_true_byHits_PDG"] == self.pdg_dict[pdg], axis=1)

            return events

    @staticmethod
    def mask_daughter_momentum(events, momentum_threshold, pdg_select):
        """
        :param events:
        :param momentum_threshold: In GeV!
        :param pdg_select:
        :return:
        """
        if momentum_threshold > 10.:
            print("Momentum threshold of ", momentum_threshold, " [GeV] is probably incorrect! Must be in GeV.")
            raise ValueError
        momentum_var = "true_beam_daughter_startP"
        daughter_pdg_var = "true_beam_daughter_PDG"
        # Momentum in GeV
        mom_mask = (events[momentum_var] > momentum_threshold) & (events[daughter_pdg_var] == pdg_select)
        return np.count_nonzero(mom_mask, axis=1)

    @staticmethod
    def single_charge_exchange(events, piplus, piminus):
        return (piplus == 0) & (piminus == 0) & (events["true_daughter_nPi0"] == 1) & (events["true_daughter_nKaon"] < 1)
               #((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def double_charge_exchange(events, piplus, piminus):
        return (piminus == 1) & (piplus == 0) & (events["true_daughter_nPi0"] == 0) & (events["true_daughter_nKaon"] < 1)
               #((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def absorption(events, piplus, piminus):
        return (piminus == 0) & (piplus == 0) & (events["true_daughter_nPi0"] == 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0)) & (events["true_daughter_nKaon"] < 1)

    @staticmethod
    def quasi_elastic(events, piplus, piminus):
        return (piminus == 0) & (piplus == 1) & (events["true_daughter_nPi0"] == 0) & \
               (events["true_daughter_nProton"] > 0)

    @staticmethod
    def pion_production(events, piplus, piminus):
        return (((piminus > 1) | (piplus > 1)) |
                ((piminus > 0) & (piplus > 0))) & (events["true_daughter_nPi0"] == 0) & \
                ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def pion_production_v2(events, piplus, piminus):
        # return (piminus + piplus + (events["true_daughter_nPi0"] == 0)) > 1
        return ( (((piminus > 0) | (piplus > 0)) & events["true_daughter_nPi0"] > 0) |
                 ((piminus == 0) & (piplus == 0) & events["true_daughter_nPi0"] > 1) |
                 ((piminus > 1) & (piplus == 0)) |
                 ((piminus == 0) & (piplus > 1)) |
                 ((piminus == 1) & (piplus == 1))
              )

    @staticmethod
    def pi0_production(events, piplus, piminus):
        return (piminus == 0) & (piplus == 0) & (events["true_daughter_nPi0"] > 1)

    @staticmethod
    def pi0_and_pion(events, piplus, piminus):
        return ((piminus > 0) | (piplus > 0)) & (events["true_daughter_nPi0"] > 0) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

    @staticmethod
    def charged_neutral_pion_production(events, piplus, piminus):
        return ((piminus > 1) | (piplus > 1)) | ((piminus > 0) & (piplus > 0)) | \
               (events["true_daughter_nPi0"] > 1)

    @staticmethod
    def mctruth_charged_neutral_pion(events, piplus, piminus):
        pi_minus_count = (events["mcreco-pi-minus-count"] == 0) & (piminus == 1)
        pi_plus_count = (events["mcreco-pi-plus-count"] == 0) & (piplus == 1)
        return (pi_minus_count | pi_plus_count) & (events["true_daughter_nPi0"] == 1)

    @staticmethod
    def mcreco_charged_neutral_pion(events):
        return ((events["mcreco-pi-minus-count"] == 1) | (events["mcreco-pi-plus-count"] == 1)) & \
               (events["true_daughter_nPi0"] == 1)

    @staticmethod
    def daughter_one_pi0_bkgd(events, piplus, piminus):
        return ((piplus > 0) | (piminus > 0)) & (events["true_daughter_nPi0"] == 1)

    @staticmethod
    def daughter_n_pi0_bkgd(events, piplus, piminus):
        return (events["true_daughter_nPi0"] > 1)

    @staticmethod
    def daughter_charged_pion_prod(events, piplus, piminus):
        return ((piminus > 0) | (piplus > 0)) & (events["true_daughter_nPi0"] == 0)


