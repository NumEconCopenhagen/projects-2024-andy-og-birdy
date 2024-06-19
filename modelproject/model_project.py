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
        par.Nw = 100     # number of workers
        par.Nc = 10      # number of capitalists
        par.delta = 0.01 # scaling factor for pollution disutility

        # b. solution
        sol = self.sol = SimpleNamespace()
        sol.p = 1 # output price
        sol.w = 1 # wage

    def pollution_disutility(self, l_total):
        """Calculate pollution disutility based on total labor"""
        par = self.par
        return par.delta * l_total ** par.beta

    def utility_w(self, c, l, total_labor=None):
        """Utility of workers with pollution impact"""
        if total_labor is None:
            total_labor = l
        return np.log(c + self.par.kappa) - self.par.omega * l ** self.par.eta - self.pollution_disutility(total_labor)

    def utility_c(self, c, l, total_labor=None):
        """Utility of capitalists with pollution impact"""
        if total_labor is None:
            total_labor = l
        return np.log(c + self.par.kappa) - self.par.omega * l ** self.par.eta - self.pollution_disutility(total_labor)

    def utility_without_pollution(self, c, l):
        """Calculate utility without pollution disutility"""
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta

    def total_pollution_disutility(self, l_w, l_c):
        """Calculate total pollution disutility from combined labor inputs"""
        par = self.par
        total_labor = self.par.Nw * l_w + self.par.Nc * l_c
        return par.delta * total_labor ** par.beta

    def social_welfare(self, l_w, l_c):
        """Calculate social welfare by summing utilities and subtracting pollution disutility"""
        p = self.sol.p
        w = self.sol.w
        pi = self.sol.pi

        c_w = (w * l_w) / p
        c_c = (w * l_c + pi) / p

        # Calculate utilities without pollution
        u_w = self.utility_without_pollution(c_w, l_w)
        u_c = self.utility_without_pollution(c_c, l_c)

        # Calculate total pollution disutility
        total_pollution = self.total_pollution_disutility(l_w, l_c)

        # Sum utilities and subtract total pollution disutility
        return self.par.Nw * u_w + self.par.Nc * u_c - total_pollution

    def workers(self):
        """Maximize utility for workers"""
        sol = self.sol
        p = sol.p
        w = sol.w

        # Solve
        obj = lambda l: -self.utility_w((w * l) / p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # Save results
        sol.l_w_star = res.x
        sol.c_w_star = (w * sol.l_w_star) / p
        sol.utility_w = -res.fun

    def capitalists(self):
        """Maximize utility for capitalists"""
        sol = self.sol
        p = sol.p
        w = sol.w
        pi = sol.pi

        # Solve
        obj = lambda l: -self.utility_c((w * l + pi) / p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # Save results
        sol.l_c_star = res.x
        sol.c_c_star = (w * sol.l_c_star + pi) / p
        sol.utility_c = -res.fun

    def firm(self):
        """Maximize firm profits"""
        par = self.par
        sol = self.sol
        p = sol.p
        w = sol.w

        # Solve
        f = lambda l: l ** par.alpha
        obj = lambda l: -(p * f(l) - w * l)
        x0 = [0.0]
        res = optimize.minimize(obj, x0, bounds=((0, None),), method='L-BFGS-B')

        # Save results
        sol.l_star = res.x[0]
        sol.y_star = f(sol.l_star)
        sol.Pi = p * sol.y_star - w * sol.l_star

    def evaluate_equilibrium(self):
        """Evaluate equilibrium"""
        par = self.par
        sol = self.sol

        # Optimal behavior of firm
        self.firm()
        sol.pi = sol.Pi / (par.Nc if par.Nc != 0 else 1)

        # Optimal behavior of households
        self.workers()
        self.capitalists()

        # Calculate total labor and pollution
        sol.total_labor = par.Nw * sol.l_w_star + par.Nc * sol.l_c_star
        sol.pollution_level = self.pollution_disutility(sol.total_labor)

        # Market clearing
        sol.goods_mkt_clearing = par.Nw * sol.c_w_star + par.Nc * sol.c_c_star - sol.y_star
        sol.labor_mkt_clearing = sol.total_labor - sol.l_star

    def find_equilibrium(self):
        """Find equilibrium"""
        par = self.par
        sol = self.sol

        # Objective function
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
        u_w = self.utility_w(sol.c_w_star, sol.l_w_star, sol.total_labor)
        u_c = self.utility_c(sol.c_c_star, sol.l_c_star, sol.total_labor)
        print(f'Workers: c = {sol.c_w_star:6.4f}, l = {sol.l_w_star:6.4f}, u = {u_w:7.4f}')
        print(f'Capitalists: c = {sol.c_c_star:6.4f}, l = {sol.l_c_star:6.4f}, u = {u_c:7.4f}')
        print(f'Goods market: {sol.goods_mkt_clearing:.8f}')
        print(f'Labor market: {sol.labor_mkt_clearing:.8f}')
        print(f'Total pollution: {sol.pollution_level:.4f}')

    def social_planner(self):
        """Social planner optimization for maximizing social welfare"""
        sol = self.sol
        p = sol.p
        w = sol.w

        # Define objective function for social planner
        obj = lambda l: -self.social_welfare(l[0], l[1])
        bounds = [(0, 1), (0, 1)]
        x0 = [sol.l_w_star, sol.l_c_star]

        # Find optimal labor supply for workers and capitalists
        res = optimize.minimize(obj, x0, bounds=bounds, method='L-BFGS-B')

        # Save results
        sol.l_w_planner = res.x[0]
        sol.l_c_planner = res.x[1]
        sol.c_w_planner = (w * sol.l_w_planner) / p
        sol.c_c_planner = (w * sol.l_c_planner + sol.pi) / p
        sol.utility_w_planner = self.utility_without_pollution(sol.c_w_planner, sol.l_w_planner)
        sol.utility_c_planner = self.utility_without_pollution(sol.c_c_planner, sol.l_c_planner)
        sol.social_welfare = self.social_welfare(sol.l_w_planner, sol.l_c_planner)

        # Calculate and print real utilities with pollution
        total_labor_planner = sol.l_w_planner * self.par.Nw + sol.l_c_planner * self.par.Nc
        real_utility_w = self.utility_w(sol.c_w_planner, sol.l_w_planner, total_labor_planner)
        real_utility_c = self.utility_c(sol.c_c_planner, sol.l_c_planner, total_labor_planner)

        # Print results
        print(f"Social planner results:")
        print(f"Workers: c = {sol.c_w_planner:6.4f}, l = {sol.l_w_planner:6.4f}, u = {real_utility_w:7.4f}")
        print(f"Capitalists: c = {sol.c_c_planner:6.4f}, l = {sol.l_c_planner:6.4f}, u = {real_utility_c:7.4f}")
        print(f"Social welfare: {sol.social_welfare:7.4f}")
