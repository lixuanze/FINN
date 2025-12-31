import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from absl import app
from absl import flags

flags.DEFINE_string('option_type', 'call', 'European Call / Put Option (Default European Call)')
flags.DEFINE_integer('set_seed', 413, 'training and testing seed (Default 413)')
FLAGS = flags.FLAGS

def main(argv):
    # Define possible volatility of volatility values
    volvol_values = [0.125, 0.15, 0.175]
    
    # Heston call price
    def Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda):
        p1 = p_Heston(S0, v0, K, r, q, T, kappa, theta, sigma, rho, lmbda, 1) # F1: characteristics function
        p2 = p_Heston(S0, v0, K, r, q, T, kappa, theta, sigma, rho, lmbda, 2) # F2
        return S0 * np.exp(-q*T) * p1 - K * np.exp(-r*T) * p2
    
    def Heston_put_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda):
        C = Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
        P = C - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return P
    
    def Heston_delta(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda, dS=0.0001):
        if FLAGS.option_type == "call":
            call_price1 = Heston_call_price(S0 + dS, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
            call_price2 = Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
            return (call_price1 - call_price2) / dS
        elif FLAGS.option_type == 'put':
            put_price1 = Heston_put_price(S0 + dS, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
            put_price2 = Heston_put_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
            return (put_price1 - put_price2) / dS
    
    # Heston probability
    def p_Heston(S0, v0, K, r, q, T, kappa, theta, sigma, rho, lmbda, j):
        integrand = lambda phi: np.real(np.exp(-1j * phi * np.log(K)) \
                                        * f_Heston(phi, S0, v0, T, r, q, kappa, theta, sigma, rho, lmbda, j) \
                                        / (1j * phi))
        integral = quad(integrand, 0, 100)[0]
        return 0.5 + (1 / np.pi) * integral
        
    p_Heston = np.vectorize(p_Heston)
    # Heston characteristic function
    def f_Heston(phi, S0, v0, T, r, q, kappa, theta, sigma, rho, lmbda, j):
    
        if j == 1:
            u = 0.5
            b = kappa + lmbda - rho * sigma
        else:
            u = -0.5
            b = kappa + lmbda
    
        a = kappa * theta
        d = np.sqrt((rho * sigma * phi * 1j - b)**2 - sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - rho * sigma * phi * 1j + d) / (b - rho * sigma * phi * 1j - d)
        C = (r - q) * phi * 1j * T + (a / sigma**2) \
                * ((b - rho * sigma * phi * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T))/(1 - g)))
        D = (b - rho * sigma * phi * 1j + d) / sigma**2 * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
    
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))
    
    
    # Parameters common for all simulations
    r = 0.00  # risk-free interest rate
    q = 0  # dividend rate
    v0 = 0.0225  # initial variance
    rho = -0.7  # correlation between Brownian motions
    kappa = 1.25  # mean reversion rate
    theta = 0.0225  # Long term mean of variance
    
    for sigma in volvol_values:
        # Store results in these lists
        all_results = []
    
        # Time list initialization
        t = [i / 250 for i in range(60, 121, 10)]
    
        # Iterate over different time to maturities
        for T in t:
            K_range = [i + 90 for i in range(21)]
            S0_range = np.linspace(75, 125, num=1000)
    
            for K in K_range:
                # Calculate Heston model prices and deltas for each S0 in the range
                for S0 in S0_range:
                    if FLAGS.option_type == "call":
                        option_price = Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda=0)
                        delta = Heston_delta(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda=0)
                        all_results.append({
                            'volvol': sigma, 
                            'T': T, 
                            'K': K, 
                            'S0': S0, 
                            'Call_Price': option_price, 
                            'Delta': delta
                        })
                    elif FLAGS.option_type == "put":  # Corrected line
                        option_price = Heston_put_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda=0)
                        delta = Heston_delta(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda=0)
                        all_results.append({
                            'volvol': sigma, 
                            'T': T, 
                            'K': K, 
                            'S0': S0, 
                            'Put_Price': option_price, 
                            'Delta': delta
                        })
    
        # Convert results to DataFrame and save to CSV
        df = pd.DataFrame(all_results)
        filename = f'heston_results/heston_model_results_{sigma}_{FLAGS.option_type}.csv'
        df.to_csv(filename, index=False)
    
        print(f"Results saved for volvol = {sigma}")

if __name__ == '__main__':
    app.run(main)
