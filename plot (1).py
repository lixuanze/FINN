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
    
    if not os.path.isdir('results'):
        os.makedirs('results')
    
    model = tf.keras.models.load_model('gbm_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
    
    print("Model restored.")
    model.summary()
    
    #Initialize time list
    t = []
    for i in range(60,121,10):
        t.append(i/250)
    
    option_mad_result = []
    option_mse_result = []
    delta_mad_result = []
    delta_mse_result = []
    
    # 3-D cost graph, for various K & T, K is ranging from (90,110), while T ranges from (60,120,10)
    for j in range(len(t)):
        T = t[j]
        print("Time to Maturity: ", T)
        K = []
        for i in range(21):
            K.append(i+90)
            
        if not os.path.isdir('results/TTM: ' + str(T)):
            os.makedirs('results/TTM: ' + str(T))
        
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
        ax.set_zlabel('C', fontsize=32, labelpad=15, fontweight='bold')
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        ax.set_title('Estimated Call Option Price for Given S and K', fontsize=40, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        plt.savefig('results/TTM: ' + str(t[j]) + '{}_{}_{}_Price.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
        
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
        ax.set_title('Estimated Hedge Ratio ($\Delta$) for Given S and K', fontsize=40, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        plt.savefig('results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))

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
        ax.set_zlabel('Error of C', fontsize=32, labelpad=20, fontweight='bold')
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        ax.set_title('Error of Predicted Call Price', fontsize=40, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        plt.savefig('results/TTM: ' + str(t[j]) + '{}_{}_{}_Price_Error.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
        
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
        ax.set_title('Error of Predicted Hedge Ratio ($\Delta$)', fontsize=40, fontweight='bold')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.grid(True)
        ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
        plt.savefig('results/TTM: ' + str(t[j]) + '{}_{}_{}_Delta_Error.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
        
        option_absolute_errors = np.abs(bs_call - pred_call)
        option_squared_errors = (bs_call - pred_call) ** 2
        option_mae = np.mean(option_absolute_errors)
        option_mse = np.mean(option_squared_errors)
        option_mad_result.append(round(option_mae, 4))
        option_mse_result.append(round(option_mse, 4))
        
        delta_absolute_errors = np.abs(bs_delta - pred_delta)
        delta_squared_errors = (bs_delta - pred_delta) ** 2
        delta_mae = np.mean(delta_absolute_errors)
        delta_mse = np.mean(delta_squared_errors)
        delta_mad_result.append(round(delta_mae, 4))
        delta_mse_result.append(round(delta_mse, 4))
        
    print("option_mad_result: ", option_mad_result)
    print("option_mse_result: ", option_mse_result)
    print("delta_mad_result: ", delta_mad_result)
    print("delta_mse_result: ", delta_mse_result)
    data = {
    "option_mad_result": option_mad_result,
    "option_mse_result": option_mse_result,
    "delta_mad_result": delta_mad_result,
    "delta_mse_result": delta_mse_result,
    }
    
    df = pd.DataFrame(data)
    filename = 'results_{}_{}_{}_.csv'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed)
    df.to_csv(filename, index=False)
    
if __name__ == '__main__':
    app.run(main)
    