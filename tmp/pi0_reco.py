

class Pi0Reco:
    def __init__(self):
        pass

    def calculate_pi0_kinematics(self, events):
        self.calculate_angle(events)

    def calculate_angle(self, events):
        angles = events["two_shower_angle"]
        energy = events["shower_energy"]
