
from types import SimpleNamespace
import numpy as np


class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        
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
    
    def check_market_clearing(self,p1):
        par = self.par
        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)
        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)
        return eps1,eps2
    

    def market_clearing_price(self,p1,maxitter=500):
        par = self.par
        eps = 1e-8    
        t = 0
        while True:
            eps1,eps2 = self.check_market_clearing(p1)
            if np.abs(eps1) < eps or t >= maxitter:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {eps1:14.8f}')
                break 

            p1= p1 + 0.5*eps1/par.alpha

            if t < 5 or t%25 == 0:
                print(f'{t:3d}: p1 = {p1:12.8f} -> excess demand -> {eps1:14.8f}')
            elif t == 5:
                print('   ...')
            t +=1

        return p1
        

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

    def optimal_price(self):
        def objective(p1):
            x1B, x2B = self.demand_B(p1)
            if 1 - x1B > 0 and 1 - x2B > 0:
                u = self.utility_A(1 - x1B, 1 - x2B)
                return -u  # Negative because we want to maximize utility
            else:
                return np.inf  # Penalize infeasible solutions
        
        result = minimize_scalar(objective, bounds=(0, 2), method='bounded')
        return result.x, -result.fun  # Return the optimal price and the corresponding utility
 