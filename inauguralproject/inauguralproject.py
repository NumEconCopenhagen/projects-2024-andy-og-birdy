
#Importing to be used later:

from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

#From where we are setting up the model

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1 - par.w1A
        par.w2B = 1 - par.w2A
        
    def utility_A(self,x1A,x2A):
        par = self.par
        u = x1A**par.alpha*x2A**(1-par.alpha)
        return u
    
    def utility_B(self,x1B,x2B):
        par = self.par
        u = x1B**par.beta*x2B**(1-par.beta)
        return u
    
    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha*((p1*par.w1A+par.w2A)/p1)
        x2A = (1-par.alpha)*(p1*par.w1A+par.w2A)
        return x1A,x2A
    
    def demand_B(self,p1):
        par = self.par
        x1B = par.beta*((p1*(1-par.w1A)+(1-par.w2A))/p1)
        x2B = (1-par.beta)*(p1*(1-par.w1A)+(1-par.w2A))
        return x1B,x2B

# we can now set up the market clearing alocation and and price, together with the exess product.

    def check_market_clearing(self,p1):  #finding the error markets for the two good
        par = self.par
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)
        return eps1,eps2
    

    def market_clearing_price(self,p1,maxitter=500):   #Want to find the the prise that makes the error's of the goods 0.
        par = self.par
        eps = 1e-8    
        t = 0
        while True:
            eps1,eps2 = self.check_market_clearing(p1)  #Takes the market error for a given price.
            if np.abs(eps1) < eps or t >= maxitter:  #See if the error is small enough to satisfy the equalibirum.
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {eps1:14.8f}')
                break

            p1= p1 + 0.5*eps1/par.alpha #if not then opdate the price.

            if t < 5 or t%25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {eps1:14.8f}')  #Print first 5 guesses.
            elif t == 5:
                print('   ...')
            t +=1  #Due the same again.

        return p1
        
# the two market clearing prices are identical, allthough the first prinst steps, and the latter does not.

    def market_clearing_price_Q8(self,p1,maxitter=500):
        par = self.par
        eps = 1e-8    
        t = 0
        while True:
            eps1,eps2 = self.check_market_clearing(p1)
            if np.abs(eps1) < eps or t >= maxitter:
                break 
            p1= p1 + 0.5*eps1/par.alpha

            t +=1

        return p1

    
#The following is setting up an uptimizer, to find the optimal price: in 4.b:

    def optimal_price(self):
        par = self.par
        def objective(p1):
            x1B, x2B = self.demand_B(p1)  #Look at the demand, b have for a given price.
            u = self.utility_A(1 - x1B, 1 - x2B) # look at what A's utility becomes under a given price 

            return -u  # Negative because we want to maximize utility for A
        
        result = minimize_scalar(objective, bounds=(0, 2), method='bounded')
        return result.x, -result.fun  # Return the optimal price and the corresponding utility for A
    
    # Finding the optimal choice in 5.b:
    
    def optimal_choice_for_A(self):
        par = self.par

        def objective(x):
            x1A, x2A = x
            u_A = self.utility_A(x1A, x2A)
            u_B_initial = self.utility_B(1 - par.w1A, 1 - par.w2A)
            u_B_current = self.utility_B(1 - x1A, 1 - x2A)

            if u_B_current >= u_B_initial:  #Make sure B is at least placed equaly as good as before.
                return -u_A  # Minimize negative of utility_A (maximize utility_A)
            else:
                return np.inf  # Penalize infeasible solutions
            
        # Bounds for x1A and x2A (between 0 and 1)
        bounds = [(0, 1), (0, 1)]
        # Optimize using scipy differential_evolution, this minimizer is used do to it havging multi start capabilities, to ensure we end up in the correct max utility for A

        result = differential_evolution(objective, bounds=bounds, tol=1e-6)

        # Retrieve optimal x1A and x2A
        x1A_optimal, x2A_optimal = result.x  

        # Calculate utilities
        utility_A_max = -result.fun
        utility_B_final = self.utility_B(1 - x1A_optimal, 1 - x2A_optimal)

        return x1A_optimal, x2A_optimal, utility_A_max, utility_B_final
    
    # 6.a:
    def maximize_combined_utility(self):
        def objective(x):
            x1A, x2A = x
            u_A = self.utility_A(x1A, x2A) 
            u_B = self.utility_B(1 - x1A, 1 - x2A)  # Utility for B with remaining endowment
            return -(u_A + u_B)  # Minimize negative of combined utility 

        # Initial guess
        # Bounds for x1A and x2A (between 0 and 1)
        bounds = [(0, 1), (0, 1)]
        # Optimize again using the mulit start minimizer, ensuring to find the global maximum.

        result = differential_evolution(objective, bounds=bounds,  tol=1e-6)

        # Retrieve optimal x1A and x2A
        x1A_optimal, x2A_optimal = result.x

        # Calculate utilities
        utility_A_max = self.utility_A(x1A_optimal, x2A_optimal)
        utility_B_max = self.utility_B(1 - x1A_optimal, 1 - x2A_optimal)

        # Exstract the wanted result.
        
        return x1A_optimal, x2A_optimal, utility_A_max, utility_B_max