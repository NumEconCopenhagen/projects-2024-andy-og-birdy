from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ProductionEconomy:
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
        self.par.kappa = 0.1
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
        par = self.par
        return par.A * (self.labor_demand(p)) ** par.gamma

    def profits(self, p):
        """Calculates the implied profits for given price."""
        par = self.par
        return (1 - par.gamma) / par.gamma * par.w * self.labor_demand(p)

    def consumption_1(self, ell, p1, p2):
        """Calculates optimal consumption for good 1."""
        par = self.par
        income = par.w * ell + par.T + self.profits(p1) + self.profits(p2)
        return par.alpha * income / p1

    def consumption_2(self, ell, p1, p2):
        """Calculates optimal consumption for good 2."""
        par = self.par
        income = par.w * ell + par.T + self.profits(p1) + self.profits(p2)
        return (1 - par.alpha) * income / (p2 + par.tau)

    def utility(self, ell, p1, p2):
        """Computes utility for given labor and prices."""
        c1 = self.consumption_1(ell, p1, p2)
        c2 = self.consumption_2(ell, p1, p2)
        par = self.par
        return np.log(c1 ** par.alpha * c2 ** (1 - par.alpha)) - par.nu * ell ** (1 + par.epsilon) / (1 + par.epsilon)

    def optimal_labor(self, p1, p2):
        """Finds the optimal labor supply by maximizing utility."""
        result = optimize.minimize_scalar(lambda ell: -self.utility(ell, p1, p2), bounds=(1e-6, 100), method='bounded')
        return result.x

    def market_clearing_conditions(self, prices):
        """Defines the market clearing conditions for labor and good market 1."""
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
        
        return [labor_market_clearing, good_market_1_clearing]

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

    def social_welfare(self, tau):
        """Calculate the social welfare function."""
        par = self.par
        sol = self.sol

        # Set the tax parameter
        par.tau = tau

        # Recalculate equilibrium with the new tau
        self.find_equilibrium_prices()

        # Recalculate T based on equilibrium consumption of good 2
        par.T = tau * sol.c2_star

        # Calculate utility
        ell_star = self.optimal_labor(sol.p1, sol.p2)
        utility = self.utility(ell_star, sol.p1, sol.p2)

        # Calculate social welfare
        swf = utility - par.kappa * sol.y2_star
        return swf

    def maximize_social_welfare(self):
        """Find the optimal value of tau to maximize social welfare."""

        # Objective function to minimize (negative of social welfare)
        def objective(tau):
            return -self.social_welfare(tau[0])

        # Callback function to print tau at each iteration
        def callback(tau):
            print(f'Current tau: {tau[0]:6.4f}')

        # Initial guess for tau
        initial_guess = [0.1]
        
        # Optimization bounds for tau
        bounds = [(0, 1)]

        # Use optimization to find the best tau
        result = optimize.minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', callback=callback)
        
        if result.success:
            tau_opt = result.x[0]
            par = self.par
            par.tau = tau_opt
            self.find_equilibrium_prices()
            par.T = tau_opt * self.sol.c2_star
            print(f'\nOptimal tau: {tau_opt:6.4f}')
            print(f'Optimal T: {par.T:6.4f}')
        else:
            raise ValueError("Optimization for social welfare failed.")


class BarycentricInterpolation:
    def barycentric_coordinates(A, B, C, y):
        """
        Calculate the barycentric coordinates of point y with respect to triangle ABC.
        
        Parameters:
        A, B, C : array-like
            Coordinates of the triangle vertices.
        y : array-like
            Coordinates of the point to be interpolated.
        
        Returns:
        r1, r2, r3 : float
            Barycentric coordinates of the point y.
        """
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def is_inside_triangle(r1, r2, r3):
        """
        Check if a point is inside a triangle based on its barycentric coordinates.
        
        Parameters:
        r1, r2, r3 : float
            Barycentric coordinates of the point.
        
        Returns:
        bool
            True if the point is inside the triangle, False otherwise.
        """
        return (0 <= r1 <= 1) and (0 <= r2 <= 1) and (0 <= r3 <= 1)

    def find_closest_points(X, y):
        """
        Find the four closest points to the given point y in the dataset X.
        
        Parameters:
        X : array-like, shape (n_samples, 2)
            Array of sample points.
        y : array-like
            The point to find the closest points to.
        
        Returns:
        A, B, C, D : array-like
            The four closest points to y.
        """
        distances = np.linalg.norm(X - y, axis=1)
        A = X[np.argmin(distances)]
        B = X[np.argsort(distances)[1]]
        C = X[np.argsort(distances)[2]]
        D = X[np.argsort(distances)[3]]
        return A, B, C, D

    def interpolate(X, f_values, y):
        """
        Interpolate the value at point y using barycentric interpolation.
        
        Parameters:
        X : array-like, shape (n_samples, 2)
            Array of sample points.
        f_values : array-like, shape (n_samples,)
            Function values at the sample points.
        y : array-like
            The point to interpolate the value at.
        
        Returns:
        float
            Interpolated value at point y.
        """
        A, B, C, D = BarycentricInterpolation.find_closest_points(X, y)
        
        # Calculate barycentric coordinates with respect to triangle ABC
        r1, r2, r3 = BarycentricInterpolation.barycentric_coordinates(A, B, C, y)
        if BarycentricInterpolation.is_inside_triangle(r1, r2, r3):
            print(f"Point y is inside triangle ABC: {r1}, {r2}, {r3}")
            f_A = f_values[np.where((X == A).all(axis=1))[0][0]]
            f_B = f_values[np.where((X == B).all(axis=1))[0][0]]
            f_C = f_values[np.where((X == C).all(axis=1))[0][0]]
            return r1 * f_A + r2 * f_B + r3 * f_C
        
        # Calculate barycentric coordinates with respect to triangle CDA
        r1, r2, r3 = BarycentricInterpolation.barycentric_coordinates(C, D, A, y)
        if BarycentricInterpolation.is_inside_triangle(r1, r2, r3):
            print(f"Point y is inside triangle CDA: {r1}, {r2}, {r3}")
            f_C = f_values[np.where((X == C).all(axis=1))[0][0]]
            f_D = f_values[np.where((X == D).all(axis=1))[0][0]]
            f_A = f_values[np.where((X == A).all(axis=1))[0][0]]
            return r1 * f_C + r2 * f_D + r3 * f_A
        
        # If y is not inside either triangle, return NaN
        print("Point y is not inside either triangle")
        return np.nan