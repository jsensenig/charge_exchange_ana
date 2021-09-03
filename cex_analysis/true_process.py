

class TrueProcess:

    def __init__(self):
        pass

    def classify_event_process(self, events):
        """
        Classify events by interaction process. Add one column of booleans per process.
        :param events: Record of events
        :return: Same Record of events but with one additional column per process
        """
        pion_inelastic = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+Inelastic")

        # Pion elastic
        events["pion_elastic"] = (events["true_beam_PDG"] == 211) & (events["true_beam_endProcess"] == "pi+elastic")

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
    Below are the definitions for all the classified processes
    '''

    @staticmethod
    def single_charge_exchange(events):
        return (events["true_daughter_nPiMinus"] == 0) & (events["true_daughter_nPiPlus"] == 0) & \
               (events["true_daughter_nPi0"] == 1) & \
               ((events["true_daughter_nProton"] > 0) | (events["true_daughter_nNeutron"] > 0))

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

