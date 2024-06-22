# economy_model.py
from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class ProductionEconomy():
    def __init__(self):
        self.par = SimpleNamespace()

        # a. parameters
        self.par.A = 1.0
        self.par.gamma = 0.5
        self.par.alpha = 0.3
        self.par.nu = 1.0
        self.par.epsilon = 2.0
        self.par.tau = 0.0
        self.par.T = 0.0
        self.par.w = 1.0

        # b. solution
        self.sol = SimpleNamespace()
        self.sol.p1 = 1.0
        self.sol.p2 = 1.0

    def labor_demand(self, p):
        """Calculates optimal labor demand for given price."""
        par = self.par
        return (p * par.A * par.gamma / par.w) ** (1 / (1 - par.gamma))

    def output(self, p):
        """Calculates output based on optimal labor demand."""
        return self.par.A * (self.labor_demand(p)) ** self.par.gamma

    def profits(self, p):
        """Calculates the implied profits for given price."""
        return (1 - self.par.gamma) / self.par.gamma * self.par.w * self.labor_demand(p)

    def consumption_1(self, ell, p1, p2):
        """Calculates optimal consumption for good 1."""
        income = self.par.w * ell + self.par.T + self.profits(p1) + self.profits(p2)
        return self.par.alpha * income / p1

    def consumption_2(self, ell, p1, p2):
        """Calculates optimal consumption for good 2."""
        income = self.par.w * ell + self.par.T + self.profits(p1) + self.profits(p2)
        return (1 - self.par.alpha) * income / (p2 + self.par.tau)

    def utility(self, ell, p1, p2):
        """Computes utility for given labor and prices."""
        c1 = self.consumption_1(ell, p1, p2)
        c2 = self.consumption_2(ell, p1, p2)
        return np.log(c1 ** self.par.alpha * c2 ** (1 - self.par.alpha)) - self.par.nu * ell ** (1 + self.par.epsilon) / (1 + self.par.epsilon)

    def optimal_labor(self, p1, p2):
        """Finds the optimal labor supply by maximizing utility."""
        result = optimize.minimize_scalar(lambda ell: -self.utility(ell, p1, p2), bounds=(1e-6, 100), method='bounded')
        return result.x

    def market_clearing_conditions(self, prices):
        """Defines the market clearing conditions for labor and goods markets."""
        p1, p2 = prices
        ell_star = self.optimal_labor(p1, p2)
        ell1_star = self.labor_demand(p1)
        ell2_star = self.labor_demand(p2)
        y1_star = self.output(p1)
        c1_star = self.consumption_1(ell_star, p1, p2)
        c2_star = self.consumption_2(ell_star, p1, p2)

        # Labor market: ell_star should equal ell1_star + ell2_star
        labor_market_clearing = ell_star - (ell1_star + ell2_star)
        
        # Good market 1: c1_star should equal y1_star
        good_market_1_clearing = c1_star - y1_star

        # Good market 2: c2_star should equal y2_star
        good_market_2_clearing = c2_star - self.output(p2)
        
        return [labor_market_clearing, good_market_1_clearing, good_market_2_clearing]

    def find_equilibrium_prices(self):
        """Find equilibrium prices p1 and p2."""
        par = self.par
        sol = self.sol

        # Define objective function
        def obj(prices):
            return self.market_clearing_conditions(prices)

        # Initial guess for prices
        initial_guess = [1.0, 1.0]
        
        # Find equilibrium prices using optimization
        res = optimize.root(obj, initial_guess, method='hybr')
        if res.success:
            sol.p1, sol.p2 = res.x
            # Evaluate equilibrium with the found prices
            ell_star = self.optimal_labor(sol.p1, sol.p2)
            sol.c1_star = self.consumption_1(ell_star, sol.p1, sol.p2)
            sol.c2_star = self.consumption_2(ell_star, sol.p1, sol.p2)
            sol.y1_star = self.output(sol.p1)
            sol.y2_star = self.output(sol.p2)
            sol.labor_mkt_clearing = ell_star - (self.labor_demand(sol.p1) + self.labor_demand(sol.p2))
            sol.goods_mkt_1_clearing = sol.c1_star - sol.y1_star
            sol.goods_mkt_2_clearing = sol.c2_star - sol.y2_star

            # Show results
            print(f'Equilibrium prices: p1 = {sol.p1:6.4f}, p2 = {sol.p2:6.4f}')
            print(f'Labor market clearing: {sol.labor_mkt_clearing:.8f}')
            print(f'Good market 1 clearing: {sol.goods_mkt_1_clearing:.8f}')
            print(f'Good market 2 clearing: {sol.goods_mkt_2_clearing:.8f}')
        else:
            raise ValueError("Equilibrium prices could not be found.")

