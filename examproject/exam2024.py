# economy_model.py
import numpy as np
from scipy.optimize import minimize_scalar

class Production_Economy:
    def __init__(self, A=1.0, gamma=0.5, alpha=0.3, nu=1.0, epsilon=2.0, tau=0.0, T=0.0, w=1.0):
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = w

    def labor_demand(self, p):
        return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def output(self, p):
        ell_star = self.labor_demand(p)
        return self.A * (ell_star) ** self.gamma

    def profits(self, p):
        ell_star = self.labor_demand(p)
        return (1 - self.gamma) / self.gamma * self.w * ell_star

    def consumption_1(self, ell, p1, p2):
        income = self.w * ell + self.T + self.profits(p1) + self.profits(p2)
        return self.alpha * income / p1

    def consumption_2(self, ell, p1, p2):
        income = self.w * ell + self.T + self.profits(p1) + self.profits(p2)
        return (1 - self.alpha) * income / (p2 + self.tau)

    def utility(self, ell, p1, p2):
        c1 = self.consumption_1(ell, p1, p2)
        c2 = self.consumption_2(ell, p1, p2)
        return np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * ell ** (1 + self.epsilon) / (1 + self.epsilon)

    def optimal_labor(self, p1, p2):
        result = minimize_scalar(lambda ell: -self.utility(ell, p1, p2), bounds=(1e-6, 100), method='bounded')
        return result.x

    def check_market_clearing(self, p1, p2):
        ell_star = self.optimal_labor(p1, p2)
        ell1_star = self.labor_demand(p1)
        ell2_star = self.labor_demand(p2)
        y1_star = self.output(p1)
        y2_star = self.output(p2)
        c1_star = self.consumption_1(ell_star, p1, p2)
        c2_star = self.consumption_2(ell_star, p1, p2)
        labor_market = np.isclose(ell_star, ell1_star + ell2_star)
        good_market_1 = np.isclose(c1_star, y1_star)
        good_market_2 = np.isclose(c2_star, y2_star)
        return labor_market and good_market_1 and good_market_2
