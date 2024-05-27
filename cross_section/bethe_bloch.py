import numpy as np
import ROOT


class BetheBloch:

    def __init__(self, mass, charge):

        self.mass = mass
        self.charge = charge

        self.K = 0.307
        self.rho = 1.396
        self.Z = 18
        self.A = 39.948
        self.I = 1.e-6 * 10.5 * 18 # MeV
        self.me = 0.511 # MeV me * c ^ 2
        self.ke_map = {}
        self.dedx_int_steps = 20

    def ke_at_length(self, ke0, track_len):

        if self.ke_map.get(int(ke0)) is None:
            self.create_spline_at_ke(ke0=int(ke0))

        delta_e = self.ke_map[int(ke0)].Eval(track_len)

        if delta_e < 0 or (ke0 - delta_e) < 0:
            return 0

        return ke0 - delta_e

    def mean_dedx(self, ke):
        """
        Calculate the mean dE/dx for a particle in LAr
        """
        gamma = (ke + self.mass) / self.mass
        beta = np.sqrt(1. - 1. / (gamma * gamma))

        wmax = 2. * self.me * beta*beta * gamma*gamma / \
               (1. + 2. * gamma * self.me / self.mass + (self.me*self.me) / (self.mass*self.mass))

        dedx = ((self.rho * self.K * self.Z * self.charge*self.charge) / (self.A * beta*beta)) * (
                    0.5 * np.log(2. * self.me * gamma*gamma * beta*beta * wmax / (self.I*self.I)) - beta*beta
                    - 0.5 * self.density_effect(beta=beta, gamma=gamma))

        return dedx

    def integrated_dedx(self, ke0, ke1):

        ke0, ke1 = (ke1, ke0) if ke0 > ke1 else (ke0, ke1)

        step_size = (ke1 - ke0) / self.dedx_int_steps

        int_step = np.arange(self.dedx_int_steps)
        dedx = self.mean_dedx(ke0 + (int_step + 0.5) * step_size)

        area = np.sum((1. / dedx[dedx > 0]) * step_size)

        return area

    def create_spline_at_ke(self, ke0):

        num_points = int(ke0 / 10)

        delta_e = []
        track_len = []
        if num_points > 1:
            for pt in range(num_points):
                ke = ke0 - pt * 10
                delta_e.append(ke0 - ke)
                track_len.append(self.integrated_dedx(ke0=ke, ke1=ke0))
        else:
            delta_e = [0, ke0]
            track_len = [0, self.integrated_dedx(ke0=0, ke1=ke0)]

        self.ke_map[ke0] = ROOT.TSpline3("KE", np.asarray(track_len, dtype='d'), np.asarray(delta_e, dtype='d'),
                                         num_points, "b2e2", 0, 0)

    @staticmethod
    def density_effect(beta, gamma):

        lar_C = 5.215
        lar_x0 = 0.201
        lar_x1 = 3.
        lar_a = 0.196
        lar_k = 3.
        x = np.log10(beta * gamma)

        # Vectorize the if/else if, the else/default condition is 0
        cond_list = [x >= lar_x1, (lar_x0 <= x) & (x < lar_x1)]
        choice_list = [2. * np.log(10.) * x - lar_C,
                       2. * np.log(10.) * x - lar_C + lar_a * np.power((lar_x1 - x), lar_k)]

        return np.select(cond_list, choice_list)
