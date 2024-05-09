from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class ProductionEconomyWithExternalityClass():
    def __init__(self):
        par = self.par = SimpleNamespace()

        # a. parameters
        par.kappa = 0.1  # home production
        par.omega = 10   # disutility of labor supply factor
        par.eta = 1.50   # curvature of disutility of labor supply
        par.alpha = 0.50 # curvature of production function
        par.beta = 0.75  # curvature of pollution disutility
        par.Nw = 0       # number of workers
        par.Nc = 0       # number of capitalists
        par.delta = 0    # scaling factor for pollution disutility

        # b. solution
        sol = self.sol = SimpleNamespace()
        sol.p = 1 # output price
        sol.w = 1 # wage

    def pollution_disutility(self, l_total):
        """ Calculate pollution disutility based on total labor """
        par = self.par
        return par.delta * l_total ** par.beta

    def utility_w(self, c, l):
        """ utility of workers with pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta - self.pollution_disutility(l)

    def utility_c(self, c, l):
        """ utility of capitalists with pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta - self.pollution_disutility(l)

    def workers(self):
        """ maximize utility for workers """
        sol = self.sol
        p = sol.p
        w = sol.w

        # a. solve
        obj = lambda l: -self.utility_w((w*l)/p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # b. save
        sol.l_w_star = res.x
        sol.c_w_star = (w * sol.l_w_star) / p
        sol.utility_w = -res.fun

    def capitalists(self):
        """ maximize utility for capitalists """
        sol = self.sol
        p = sol.p
        w = sol.w
        pi = sol.pi

        # a. solve
        obj = lambda l: -self.utility_c((w*l + pi)/p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # b. save
        sol.l_c_star = res.x
        sol.c_c_star = (w * sol.l_c_star + pi) / p
        sol.utility_c = -res.fun

    def firm(self):
        """ maximize firm profits """
        par = self.par
        sol = self.sol
        p = sol.p
        w = sol.w

        # a. solve
        f = lambda l: l ** par.alpha
        obj = lambda l: -(p * f(l) - w * l)
        x0 = [0.0]
        res = optimize.minimize(obj, x0, bounds=((0, None),), method='L-BFGS-B')

        # b. save
        sol.l_star = res.x[0]
        sol.y_star = f(sol.l_star)
        sol.Pi = p * sol.y_star - w * sol.l_star

    def evaluate_equilibrium(self):
        """ evaluate equilibrium """
        par = self.par
        sol = self.sol

        # a. optimal behavior of firm
        self.firm()
        sol.pi = sol.Pi / par.Nc

        # b. optimal behavior of households
        self.workers()
        self.capitalists()

        # c. calculate total labor and pollution
        sol.total_labor = par.Nw * sol.l_w_star + par.Nc * sol.l_c_star
        sol.pollution_level = self.pollution_disutility(sol.total_labor)

        # d. market clearing
        sol.goods_mkt_clearing = par.Nw * sol.c_w_star + par.Nc * sol.c_c_star - sol.y_star
        sol.labor_mkt_clearing = sol.total_labor - sol.l_star

    def find_equilibrium(self):
        """ find equilibrium """
        par = self.par
        sol = self.sol

        # Define objective function
        def obj(w):
            sol.w = w
            self.evaluate_equilibrium()
            return sol.goods_mkt_clearing

        # Find equilibrium wage using optimization
        res = optimize.root_scalar(obj, bracket=[0.1, 1.5], method='bisect')
        sol.w = res.root

        # Evaluate equilibrium with the found wage
        self.evaluate_equilibrium()

        # Show results
        u_w = self.utility_w(sol.c_w_star, sol.l_w_star)
        u_c = self.utility_c(sol.c_c_star, sol.l_c_star)
        print(f'workers      : c = {sol.c_w_star:6.4f}, l = {sol.l_w_star:6.4f}, u = {u_w:7.4f}')
        print(f'capitalists  : c = {sol.c_c_star:6.4f}, l = {sol.l_c_star:6.4f}, u = {u_c:7.4f}')
        print(f'goods market : {sol.goods_mkt_clearing:.8f}')
        print(f'labor market : {sol.labor_mkt_clearing:.8f}')
        print(f'total pollution: {sol.pollution_level:.4f}')