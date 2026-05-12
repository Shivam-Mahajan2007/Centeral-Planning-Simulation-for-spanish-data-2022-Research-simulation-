"""
agents.py
---------
Agent classes for Multi-Agent CGE simulation.

Lightweight data containers for firms and households.
Heavy numerical work is offloaded to Julia modules.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Firm:
    """Sector-level representative firm in MA-CGE model."""
    
    sector_id: int
    K: float            # capital stock
    X: float            # gross output
    P: float            # price
    markup: float
    profit: float
    
    def compute_output(self, K: float, L: float, gamma: float, rho: float, 
                      X_leontief: float) -> float:
        """
        Compute output using CES production function.
        
        Parameters:
        -----------
        K, L : float
            Capital and labor inputs
        gamma, rho : float
            CES parameters
        X_leontief : float
            Output constraint from Leontief intermediate inputs
            
        Returns:
        --------
        X : float
            Actual output (min of CES and Leontief constraints)
        """
        # Primary factor composite
        F = (gamma * K**rho + (1 - gamma) * L**rho)**(1/rho)
        return min(X_leontief, F)
    
    def compute_mpk(self, K: float, L: float, gamma: float, rho: float, 
                   X: float) -> float:
        """
        Compute marginal product of capital.
        
        Parameters:
        -----------
        K, L : float
            Capital and labor inputs
        gamma, rho : float
            CES parameters
        X : float
            Current output
            
        Returns:
        --------
        MPK : float
            Marginal product of capital
        """
        F = (gamma * K**rho + (1 - gamma) * L**rho)**(1/rho)
        return gamma * K**(rho - 1) * F**(1 - rho)
    
    def compute_mc(self, w: float, r: float, l_coef: float, k_coef: float,
                   P_inputs: np.ndarray, A_col: np.ndarray) -> float:
        """
        Compute marginal cost.
        
        Parameters:
        -----------
        w, r : float
            Wage rate and rental rate of capital
        l_coef, k_coef : float
            Labor and capital coefficients per unit output
        P_inputs : np.ndarray
            Input prices
        A_col : np.ndarray
            Input coefficients from A-matrix
            
        Returns:
        --------
        MC : float
            Marginal cost per unit output
        """
        labor_cost = w * l_coef
        capital_cost = r * k_coef
        intermediate_cost = np.dot(P_inputs, A_col)
        return labor_cost + capital_cost + intermediate_cost
    
    def update_markup(self, demand: float, supply: float, eta: float, 
                     markup_max: float) -> float:
        """
        Update markup based on demand-supply imbalance.
        
        Parameters:
        -----------
        demand, supply : float
            Realized demand and supply
        eta : float
            Markup adjustment speed
        markup_max : float
            Maximum allowed markup
            
        Returns:
        --------
        new_markup : float
            Updated markup
        """
        if supply <= 0:
            return self.markup
        
        demand_pressure = (demand / supply) - 1.0
        new_markup = self.markup + eta * demand_pressure
        return np.clip(new_markup, 0.0, markup_max)
    
    def compute_investment(self, q: float, delta: float, phi: float, 
                          K: float) -> float:
        """
        Compute investment using Tobin-q accelerator rule.
        
        Parameters:
        -----------
        q : float
            Tobin's q ratio
        delta : float
            Depreciation rate
        phi : float
            Investment sensitivity parameter
        K : float
            Current capital stock
            
        Returns:
        --------
        I : float
            Investment amount
        """
        replacement_investment = delta * K
        net_investment = phi * K * max(q - 1.0, 0.0)
        return replacement_investment + net_investment


@dataclass
class Household:
    """CRRA utility maximizing household in MA-CGE model."""
    
    hh_id: int
    Y: float            # income
    alpha: np.ndarray   # (n,) preference weights
    sigma: np.ndarray   # (n,) CRRA exponents
    
    def demand(self, P: np.ndarray, Y: float) -> np.ndarray:
        """
        Compute Marshallian demand given prices and income.
        
        Parameters:
        -----------
        P : np.ndarray
            Price vector
        Y : float
            Household income
            
        Returns:
        --------
        C : np.ndarray
            Consumption bundle
        """
        # Compute budget constraint multiplier
        price_term = self.alpha**(1.0/self.sigma) * P**(1.0 - 1.0/self.sigma)
        mu_h = np.sum(price_term) / Y
        
        # Marshallian demands
        C = (self.alpha / (mu_h * P))**(1.0/self.sigma)
        return C
    
    def indirect_utility(self, P: np.ndarray, Y: float) -> float:
        """
        Compute indirect utility given prices and income.
        
        Parameters:
        -----------
        P : np.ndarray
            Price vector
        Y : float
            Household income
            
        Returns:
        --------
        U : float
            Indirect utility
        """
        # Price index for CRRA preferences
        price_term = self.alpha**(1.0/self.sigma) * P**(1.0 - 1.0/self.sigma)
        price_index = np.sum(price_term)**(self.sigma[0]/(self.sigma[0] - 1.0))
        
        # Indirect utility
        U = Y**(1.0 - self.sigma[0]) / ((1.0 - self.sigma[0]) * price_index)
        return U
    
    def expenditure_function(self, P: np.ndarray, U: float) -> float:
        """
        Compute expenditure needed to achieve utility level U.
        
        Parameters:
        -----------
        P : np.ndarray
            Price vector
        U : float
            Target utility level
            
        Returns:
        --------
        Y : float
            Required expenditure
        """
        # Price index for CRRA preferences
        price_term = self.alpha**(1.0/self.sigma) * P**(1.0 - 1.0/self.sigma)
        price_index = np.sum(price_term)**(self.sigma[0]/(self.sigma[0] - 1.0))
        
        # Expenditure function
        Y = ((1.0 - self.sigma[0]) * U * price_index)**(1.0/(1.0 - self.sigma[0]))
        return Y
