import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as si
from scipy.integrate import quad

from absl import app
from absl import flags

flags.DEFINE_string('option_type', 'call', 'European Call / Put Option (Default European Call)')
flags.DEFINE_integer('set_seed', 413, 'training and testing seed (Default 413)')
flags.DEFINE_float('volvol', 0.15, 'volatility of volatility (Default 15%)')
FLAGS = flags.FLAGS

def main(argv):
    # Parameters
    r = 0.00    # risk-free interest rate
    q = 0    # dividend rate
    v0 = 0.0225 # initial variance
    rho = -0.7  # correlation between Brownian motions
    kappa = 1.25   # mean reversion rate
    theta = 0.0225 # Long term mean of variance
    sigma = FLAGS.volvol  # volatility of volatility
    lmbda = 0    # market price of volatility risk
    
    # Option values
    # Heston call price
    def Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda):
        p1 = p_Heston(S0, v0, K, r, q, T, kappa, theta, sigma, rho, lmbda, 1) # F1: characteristics function
        p2 = p_Heston(S0, v0, K, r, q, T, kappa, theta, sigma, rho, lmbda, 2) # F2
        return S0 * np.exp(-q*T) * p1 - K * np.exp(-r*T) * p2
    
    def Heston_delta(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda, dS=0.0001):
        call_price1 = Heston_call_price(S0 + dS, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
        call_price2 = Heston_call_price(S0, v0, K, T, r, q, kappa, theta, sigma, rho, lmbda)
        return (call_price1 - call_price2) / dS
    
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
    
    if not os.path.isdir('heston_results'):
        os.makedirs('heston_results')
    
    if FLAGS.option_type == "call":
        model = tf.keras.models.load_model('heston_call_trained_models/heston_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
    elif FLAGS.option_type == "put":
        model = tf.keras.models.load_model('heston_put_trained_models/heston_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
        
    heston_results = pd.read_csv('heston_results/heston_model_results_{}_{}.csv'.format(FLAGS.volvol, FLAGS.option_type))
    
    print("Model restored.")
    model.summary()
    
    #Initialize time list
    t = []
    for i in range(60,121,10):
        t.append(i/250)
    
    option_rmad_result = []
    option_rmse_result = []
    option_nmad_result = []
    option_mse_result  = []
    
    delta_rmad_result = []
    delta_rmse_result = []
    delta_nmad_result = []
    delta_mse_result  = []

    
    # 3-D cost graph, for various K & T, K is ranging from (90,110), while T ranges from (60,120,10)
    for j in range(len(t)):
        T = t[j]
        print("Time to Maturity: ", T)
        K = []
        for i in range(21):
            K.append(i+90)
        
        
        X_train = np.ones((1000,3,21))
        for i in range(21):
            X_train[:, 0, i] = np.linspace(75, 125, num=1000)
            X_train[:, 1, i] *= K[i]
            X_train[:, 2, i] *= T
        bs_call = np.ones((1000,21))
        bs_delta = np.ones((1000,21))
        
        for i in range(21):
            if FLAGS.option_type == "call":
                relevant_info = heston_results[(heston_results['T'] == T) & (heston_results['K'] == K[i])][["S0", "Call_Price", "Delta"]]
                call = relevant_info["Call_Price"].values
                delta = relevant_info["Delta"].values
            elif FLAGS.option_type == "put":
                relevant_info = heston_results[(heston_results['T'] == T) & (heston_results['K'] == K[i])][["S0", "Put_Price", "Delta"]]
                call = relevant_info["Put_Price"].values
                delta = relevant_info["Delta"].values
            bs_call[:,i] = call
            bs_delta[:,i] = delta
        pred_call = np.ones((1000,21))
        pred_delta = np.ones((1000,21))

		# Evaluation
        for i in range(21):
            r = tf.fill([tf.shape(input=X_train)[0],1], np.float64(0.00), name = 'r') # interest rate, if applicable
            S = tf.slice(X_train[:,:,i], (0,0), (-1,1))
            K = tf.slice(X_train[:,:,i], (0,1), (-1,1))
            T = tf.slice(X_train[:,:,i], (0,2), (-1,1))
            with tf.GradientTape() as tape1:
                tape1.watch([S,T])
                with tf.GradientTape() as tape2:
                    tape2.watch([S, T])
                    X = tf.concat([S/(K*tf.exp(-r*T)), T], 1) # input matrix for ANN

                    out = model(X)
                    if FLAGS.option_type == 'call':
                        out_values = K*tf.where(tf.greater(T, 1e-3), out, tf.maximum(S/K - 1, 0))
                    elif FLAGS.option_type == 'put':
                        out_values = K*tf.where(tf.greater(T, 1e-3), out, tf.maximum(1 - S/K, 0))
                delta_values, theta_values = tape2.gradient(out_values, [S,T])
            gamma_values, charm_values = tape1.gradient(delta_values, [S,T])

            out_values = out_values.numpy()
            delta_values = delta_values.numpy()
            theta_values = theta_values.numpy()
            theta_values = - theta_values
            gamma_values = gamma_values.numpy()
            charm_values = charm_values.numpy()
            charm_values = - charm_values
            pred_call[:,i] = out_values.reshape(1000,)
            pred_delta[:,i] = delta_values.reshape(1000,)

        stock_price = X_train[:, 0, :].reshape(1000,21)
        strike_price = X_train[:, 1, :].reshape(1000,21)
        
        plt.figure(figsize=(15, 15), facecolor='white')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, pred_call, color='red', label='FINN')
        ax.plot_wireframe(stock_price, strike_price, bs_call, color='green', label='Heston')
        ax.tick_params(axis='x', labelsize=17.5)
        ax.tick_params(axis='y', labelsize=17.5)
        ax.tick_params(axis='z', labelsize=17.5)
        ax.set_xlabel('S', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_ylabel('K', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_zlabel('{}'.format(FLAGS.option_type[0].upper()), fontsize=32, labelpad=15, fontweight='bold')
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        plt.savefig('heston_results/TTM: ' + str(t[j]) + '{}_{}_{}_Price.png'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
        
        plt.figure(figsize=(15, 15), facecolor='white')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, pred_delta, color='red', label='FINN')
        ax.plot_wireframe(stock_price, strike_price, bs_delta, color='green', label='Heston')
        ax.tick_params(axis='x', labelsize=17.5)
        ax.tick_params(axis='y', labelsize=17.5)
        ax.tick_params(axis='z', labelsize=17.5)
        ax.set_xlabel('S', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_ylabel('K', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_zlabel('$\Delta$', fontsize=32, labelpad=15, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        plt.savefig('heston_results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta.png'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))

		# Error of Results
        plt.figure(figsize=(15, 15), facecolor='white', edgecolor='black')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, bs_call - pred_call, color='black', label='Error')
        ax.tick_params(axis='x', labelsize=17.5)
        ax.tick_params(axis='y', labelsize=17.5)
        ax.tick_params(axis='z', labelsize=17.5)
        ax.set_xlabel('S', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_ylabel('K', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_zlabel('Error of {}'.format(FLAGS.option_type[0].upper()), fontsize=32, labelpad=20, fontweight='bold')
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        plt.savefig('heston_results/TTM: ' + str(t[j]) + '{}_{}_{}_Price_Error.png'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
        
        plt.figure(figsize=(15, 15), facecolor='white', edgecolor='black')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, bs_delta - pred_delta, color='black', label='Error')
        ax.tick_params(axis='x', labelsize=17.5)
        ax.tick_params(axis='y', labelsize=17.5)
        ax.tick_params(axis='z', labelsize=17.5)
        ax.set_xlabel('S', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_ylabel('K', fontsize=32, labelpad=20, fontweight='bold')
        ax.set_zlabel('Error of $\Delta$', fontsize=32, labelpad=20, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        plt.savefig('heston_results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta_Error.png'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
        

        eps = 1e-12

        # ---------- Price metrics ----------
        err_c = bs_call - pred_call
        abs_err_c = np.abs(err_c)
        sq_err_c  = err_c**2
        
        price_floor = 1e-1  # keep your choice (or 0.01 if that's what you used later)
        mask_c = np.abs(bs_call) > price_floor
        
        # RMAD = sum|e| / sum|y|  (WAPE-style, stable; no per-point division)
        den_c = np.sum(np.abs(bs_call[mask_c])) + eps
        option_rmad = np.sum(abs_err_c[mask_c]) / den_c
        
        # MSE (keep as-is) and RMSE = sqrt(MSE)
        option_mse = np.mean(sq_err_c)
        den2_c = np.sum(bs_call[mask_c]**2) + eps
        option_rmse = np.sqrt(np.sum(sq_err_c[mask_c]) / den2_c)

        # NMAD = MAE normalized by (max-min) of true values over the same mask
        range_c = (np.max(bs_call[mask_c]) - np.min(bs_call[mask_c])) + eps
        option_nmad = np.mean(abs_err_c[mask_c]) / range_c
        
        option_rmad_result.append(round(option_rmad, 4))
        option_rmse_result.append(round(option_rmse, 4))
        option_nmad_result.append(round(option_nmad, 4))
        option_mse_result.append(round(option_mse, 4))
        
        
        # ---------- Delta metrics ----------
        err_d = bs_delta - pred_delta
        abs_err_d = np.abs(err_d)
        sq_err_d  = err_d**2
        
        delta_floor = 1e-2  # keep your choice (or 1e-3 if you used that later)
        mask_d = np.abs(bs_delta) > delta_floor
        
        den_d = np.sum(np.abs(bs_delta[mask_d])) + eps
        delta_rmad = np.sum(abs_err_d[mask_d]) / den_d
        
        delta_mse = np.mean(sq_err_d)
        den2_d = np.sum(bs_delta[mask_d]**2) + eps
        delta_rmse = np.sqrt(np.sum(sq_err_d[mask_d]) / den2_d)

        range_d = (np.max(bs_delta[mask_d]) - np.min(bs_delta[mask_d])) + eps
        delta_nmad = np.mean(abs_err_d[mask_d]) / range_d
        
        delta_rmad_result.append(round(delta_rmad, 4))
        delta_rmse_result.append(round(delta_rmse, 4))
        delta_nmad_result.append(round(delta_nmad, 4))
        delta_mse_result.append(round(delta_mse, 4))

    data = {
    "option_rmad_result": option_rmad_result,
    "option_rmse_result": option_rmse_result,
    "option_nmad_result": option_nmad_result,
    "option_mse_result":  option_mse_result,

    "delta_rmad_result": delta_rmad_result,
    "delta_rmse_result": delta_rmse_result,
    "delta_nmad_result": delta_nmad_result,
    "delta_mse_result":  delta_mse_result,
}
    
    df = pd.DataFrame(data)
    if FLAGS.option_type == "call":
        filename = 'heston_call_trained_models/heston_results_{}_{}_{}_.csv'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed)
    elif FLAGS.option_type == "put":
        filename = 'heston_put_trained_models/heston_results_{}_{}_{}_.csv'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed)
    df.to_csv(filename, index=False)
    
if __name__ == '__main__':
    app.run(main)
    