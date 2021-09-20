from dataclasses import dataclass


@dataclass
class Pi0Data:
    """
    Data class for reconstructed pi0 variables
    """

    E1:     float   # Energy of leading gamma
    E2:     float   # Energy of sub-leading gamma
    Theta1: float   # Angle (wrt to beam direction) of leading gamma
    Theta2: float   # Angle (wrt to beam direction) of sub-leading gamma
    angle:  float   # Angle (wrt to beam direction) of pi0
    p:      float   # Momentum of pi0
