from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar, root_scalar, minimize

class ProductionEconomy:
    def __init__(self):
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()

        # Parameters
        self.par.alpha = 0.50  # Curvature of production function
        self.par.omega = 10    # Disutility of labor supply factor
        self.par.eta = 1.50    # Curvature of disutility of labor supply
        self.par.kappa = 0.1   # Home production

        # Initial guesses
        self.sol.p = 1  # Price of output
        self.sol.w = 1  # Wage rate

    def utility(self, c, l):
        """ Utility function combining consumption and labor disutility """
        return np.log(c + self.par.kappa) - self.par.omega * l ** self.par.eta

    def production_function(self, l):
        """ Production function """
        return l ** self.par.alpha

    def firm_problem(self, w, p):
        """ Firm's profit maximization problem """
        obj = lambda l: -(p * self.production_function(l) - w * l)  # Negative profit for minimization
        res = minimize(obj, [0.1], bounds=[(0, None)])
        l_star = res.x[0]
        y_star = self.production_function(l_star)
        Pi = p * y_star - w * l_star
        return l_star, y_star, Pi

    def worker_problem(self, w, p):
        """ Worker's utility maximization problem """
        budget_constraint = lambda l: (w * l) / p  # c = (w * l) / p
        obj = lambda l: -self.utility(budget_constraint(l), l)  # Negative utility for minimization
        res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
        l_star = res.x
        c_star = budget_constraint(l_star)
        return l_star, c_star

    def find_equilibrium(self):
        """ Find market equilibrium by adjusting wage to clear the labor market """
        def market_clearing(w):
            l_demand, y_star, Pi = self.firm_problem(w, self.sol.p)
            l_supply, c_star = self.worker_problem(w, self.sol.p)
            return l_supply - l_demand  # Market clearing condition

        res = root_scalar(market_clearing, bracket=[0.5, 2.0], method='bisect')
        if res.converged:
            self.sol.w = res.root
            self.sol.l_star, self.sol.c_star = self.worker_problem(self.sol.w, self.sol.p)
            self.sol.l_demand, self.sol.y_star, self.sol.Pi = self.firm_problem(self.sol.w, self.sol.p)

        return self.sol.w, self.sol.c_star, self.sol.l_star, self.sol.y_star, self.sol.Pi

# Instantiate and use the model
model = ImprovedProductionEconomy()
equilibrium_wage, equilibrium_consumption, equilibrium_labor, equilibrium_output, equilibrium_profit = model.find_equilibrium()
equilibrium_wage, equilibrium_consumption, equilibrium_labor, equilibrium_output, equilibrium_profit
