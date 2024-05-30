from abc import abstractmethod


class XSecBase:

    def __init__(self):
        self.consts = {"rho": 1.1,
                       "Nav": 1.e23,
                       "mpip": 139.57,
                       "mpi0": 135.}

    @abstractmethod
    def calc_xsec(self):
        """
        API to the cross-section calculations
        """
        pass


class XSecTotal(XSecBase):
    """
    The total cross-section so just for the incoming beam particles.
    """
    def __init__(self):
        super().__init__()

    def calc_xsec(self):
        pass


class XSecDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and a single daughter variable, e.g. pi0 Ke, proton angle wrt to beam
    """
    def __init__(self):
        super().__init__()

    def calc_xsec(self):
        pass


class XSecDoubleDiff(XSecBase):
    """
    The differential cross-section so for the incoming beam particles
    and two daughter variables, e.g. pi0 Ke, pi0 angle wrt to beam
    """
    def __init__(self):
        super().__init__()

    def calc_xsec(self):
        pass

