import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as si

from absl import app
from absl import flags

flags.DEFINE_string('option_type', 'call', 'European Call / Put Option (Default European Call)')
flags.DEFINE_integer('set_seed', 413, 'training and testing seed (Default 413)')
flags.DEFINE_float('volatility', 0.15, 'stock volatility (Default 15%)')
FLAGS = flags.FLAGS

def main(argv):
    def bs(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        call_delta = si.norm.cdf(d1)
        call_theta = - (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2)
        call_gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        call_charm = - si.norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        put_delta = -si.norm.cdf(-d1)
        put_theta = - (S * sigma * si.norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * si.norm.cdf(-d2)
        put_gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        put_charm = - si.norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        if FLAGS.option_type == 'call':
            return call, call_delta, call_theta, call_gamma, call_charm
        elif FLAGS.option_type == 'put':
            return put, put_delta, put_theta, put_gamma, put_charm
    
    if not os.path.isdir('gbm_results'):
        os.makedirs('gbm_results')
    
    if FLAGS.option_type == "call":
        model = tf.keras.models.load_model('gbm_call_trained_models/gbm_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
    elif FLAGS.option_type == "put":
        model = tf.keras.models.load_model('gbm_put_trained_models/gbm_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
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
        
        X_train = np.ones((10000,3,21))
        for i in range(21):
            X_train[:, 0, i] = np.linspace(75, 125, num=10000)
            X_train[:, 1, i] *= K[i]
            X_train[:, 2, i] *= T
        bs_call = np.ones((10000,21))
        bs_delta = np.ones((10000,21))
        bs_theta = np.ones((10000,21))
        bs_gamma = np.ones((10000,21))
        bs_charm = np.ones((10000,21))
        
        for i in range(21):
            call, delta, theta, gamma, charm = bs(X_train[:,0,i], K[i], T, 0.00, FLAGS.volatility)
            bs_call[:,i] = call
            bs_delta[:,i] = delta
            bs_theta[:,i] = theta
            bs_gamma[:,i] = gamma
            bs_charm[:,i] = charm
        pred_call = np.ones((10000,21))
        pred_delta = np.ones((10000,21))
        pred_theta = np.ones((10000,21))
        pred_gamma = np.ones((10000,21))
        pred_charm = np.ones((10000,21))

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
            pred_call[:,i] = out_values.reshape(10000,)
            pred_delta[:,i] = delta_values.reshape(10000,)
            pred_theta[:,i] = theta_values.reshape(10000,)
            pred_gamma[:,i] = gamma_values.reshape(10000,)
            pred_charm[:,i] = charm_values.reshape(10000,)

        stock_price = X_train[:, 0, :].reshape(10000,21)
        strike_price = X_train[:, 1, :].reshape(10000,21)
        
        plt.figure(figsize=(15, 15), facecolor='white')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, pred_call, color='red', label='FINN')
        ax.plot_wireframe(stock_price, strike_price, bs_call, color='green', label='Black-Scholes')
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
        plt.savefig('gbm_results/TTM: ' + str(t[j]) + '{}_{}_{}_Price.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
        
        plt.figure(figsize=(15, 15), facecolor='white')
        plt.rcParams['axes.facecolor'] = 'white'
        ax = plt.axes(projection='3d')
        
        angle = 225
        ax.view_init(37.5, angle)
        ax.plot_wireframe(stock_price, strike_price, pred_delta, color='red', label='FINN')
        ax.plot_wireframe(stock_price, strike_price, bs_delta, color='green', label='Black-Scholes')
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
        plt.savefig('gbm_results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))

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
        plt.savefig('gbm_results/TTM: ' + str(t[j]) + '{}_{}_{}_Price_Error.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))

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
        plt.savefig('gbm_results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta_Error.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
        
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
        filename = 'gbm_call_trained_models/results_{}_{}_{}_.csv'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed)
    elif FLAGS.option_type == "put":
        filename = 'gbm_put_trained_models/results_{}_{}_{}_.csv'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed)
    df.to_csv(filename, index=False)
    
if __name__ == '__main__':
    app.run(main)
    