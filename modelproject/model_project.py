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
        par.Nw = 1       # number of workers
        par.Nc = 1      # number of capitalists
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
        """ Utility of workers with pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta - self.pollution_disutility(l)

    def utility_c(self, c, l):
        """ Utility of capitalists with pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta - self.pollution_disutility(l)

    def workers(self):
        """ Maximize utility for workers """
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
        """ Maximize utility for capitalists """
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
        """ Maximize firm profits """
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
        """ Evaluate equilibrium """
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
        """ Find equilibrium """
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


class ProductionEconomyWithWorkerPollution():
    def __init__(self):
        self.par = SimpleNamespace()

        # a. parameters
        self.par.kappa = 0.1  # home production
        self.par.omega = 5   # disutility of labor supply factor
        self.par.eta = 1.50   # curvature of disutility of labor supply
        self.par.alpha = 0.50 # curvature of production function
        self.par.beta = 0.75  # curvature of pollution disutility
        self.par.Nw = 1000       # number of workers
        self.par.Nc = 10      # number of capitalists
        self.par.delta = 0.1  # scaling factor for pollution disutility

        # b. solution
        self.sol = SimpleNamespace()
        self.sol.p = 1 # output price
        self.sol.w = 1 # wage

    def pollution_disutility(self, l_total):
        """ Calculate pollution disutility based on total labor """
        par = self.par
        return par.delta * l_total ** par.beta

    def utility_w(self, c, l):
        """ Utility of workers with pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta - self.pollution_disutility(l)

    def utility_c(self, c, l):
        """ Utility of capitalists without pollution impact """
        par = self.par
        return np.log(c + par.kappa) - par.omega * l ** par.eta

    def workers(self):
        """ Maximize utility for workers """
        sol = self.sol
        p = sol.p
        w = sol.w

        # a. solve
        obj = lambda l: -self.utility_w((w * l) / p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # b. save
        sol.l_w_star = res.x
        sol.c_w_star = (w * sol.l_w_star) / p
        sol.utility_w = -res.fun

    def capitalists(self):
        """ Maximize utility for capitalists """
        sol = self.sol
        p = sol.p
        w = sol.w
        pi = sol.pi

        # a. solve
        obj = lambda l: -self.utility_c((w * l + pi) / p, l)
        res = optimize.minimize_scalar(obj, bounds=(0, 1), method='bounded')

        # b. save
        sol.l_c_star = res.x
        sol.c_c_star = (w * sol.l_c_star + pi) / p
        sol.utility_c = -res.fun

    def firm(self):
        """ Maximize firm profits """
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
        """ Evaluate equilibrium """
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
        sol.pollution_level = self.pollution_disutility(sol.total_labor)  # Affecting total labor

        # d. market clearing
        sol.goods_mkt_clearing = par.Nw * sol.c_w_star + par.Nc * sol.c_c_star - sol.y_star
        sol.labor_mkt_clearing = sol.total_labor - sol.l_star

    def find_equilibrium(self):
        """ Find equilibrium """
        par = self.par
        sol = self.sol

        # Define objective function
        def obj(w):
            sol.w = w
            self.evaluate_equilibrium()
            return np.abs(sol.goods_mkt_clearing) + np.abs(sol.labor_mkt_clearing)

        # Find equilibrium wage using optimization
        res = optimize.minimize_scalar(obj, bounds=[0.1, 1.5], method='bounded')
        sol.w = res.x

        # Evaluate equilibrium with the found wage
        self.evaluate_equilibrium()

        # Show results
        u_w = self.utility_w(sol.c_w_star, sol.l_w_star)
        u_c = self.utility_c(sol.c_c_star, sol.l_c_star)
        print(f'Free Market Equilibrium')
        print(f'workers      : c = {sol.c_w_star:6.4f}, l = {sol.l_w_star:6.4f}, u = {u_w:7.4f}')
        print(f'capitalists  : c = {sol.c_c_star:6.4f}, l = {sol.l_c_star:6.4f}, u = {u_c:7.4f}')
        print(f'goods market : {sol.goods_mkt_clearing:.8f}')
        print(f'labor market : {sol.labor_mkt_clearing:.8f}')
        print(f'total pollution: {sol.pollution_level:.4f}')

    def social_planner(self):
        """ Find the social planner's equilibrium """
        par = self.par
        sol = self.sol

    # Define social welfare function
        def social_welfare(l):
            l_w, l_c = l
            y = (l_w + l_c) ** par.alpha
            pi = y - (l_w + l_c)
            c_w = (pi * l_w / (l_w + l_c)) + par.kappa
            c_c = (pi * l_c / (l_w + l_c)) + par.kappa
            total_labor = l_w + l_c
            u_w = self.utility_w(c_w, l_w)
            u_c = self.utility_c(c_c, l_c)
            pollution = self.pollution_disutility(total_labor)
            return -(u_w + u_c - pollution)  # Minimize negative welfare including pollution

        # Solve for the social planner's equilibrium
        bounds = [(0, 1), (0, 1)]
        initial_guess = [0.2, 0.2]  # Adjusted for possibly lower labor inputs
        res = optimize.minimize(social_welfare, initial_guess, bounds=bounds, method='L-BFGS-B')

        # Save results
        sol.l_w_sp, sol.l_c_sp = res.x
        y = (sol.l_w_sp + sol.l_c_sp) ** par.alpha
        pi = y - (sol.l_w_sp + sol.l_c_sp)
        sol.c_w_sp = (pi * sol.l_w_sp / (sol.l_w_sp + sol.l_c_sp)) + par.kappa
        sol.c_c_sp = (pi * sol.l_c_sp / (sol.l_w_sp + sol.l_c_sp)) + par.kappa
        sol.u_w_sp = self.utility_w(sol.c_w_sp, sol.l_w_sp)
        sol.u_c_sp = self.utility_c(sol.c_c_sp, sol.l_c_sp)
        sol.pollution_level_sp = self.pollution_disutility(sol.l_w_sp + sol.l_c_sp)

        # Show results
        print(f'Social Planner Equilibrium')
        print(f'workers      : c = {sol.c_w_sp:6.4f}, l = {sol.l_w_sp:6.4f}, u = {sol.u_w_sp:7.4f}')
        print(f'capitalists  : c = {sol.c_c_sp:6.4f}, l = {sol.l_c_sp:6.4f}, u = {sol.u_c_sp:7.4f}')
        print(f'total pollution: {sol.pollution_level_sp:.4f}')
        print(f'total pollution: {sol.pollution_level_sp:.4f}')

    def compare_equilibria(self):
        """ Compare free market and social planner equilibria """
        self.find_equilibrium()
        self.social_planner()

        # Compare social welfare
        sol = self.sol
        social_welfare_fm = sol.utility_w + sol.utility_c
        social_welfare_sp = sol.u_w_sp + sol.u_c_sp

        print(f'Social Welfare Comparison')
        print(f'Free Market  : {social_welfare_fm:7.4f}')
        print(f'Social Planner: {social_welfare_sp:7.4f}')

