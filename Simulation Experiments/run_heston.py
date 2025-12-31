import os
'''
Sources:
https://github.com/Robin-Guilliou/Option-Pricing
'''
import math
import random
import decimal
import scipy.linalg
import numpy.random as nrand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app
from absl import flags

flags.DEFINE_string('option_type', 'call', 'European Call / Put Option (Default European Call)')
flags.DEFINE_integer('set_seed', 413, 'training and testing seed (Default 413)')
flags.DEFINE_float('volvol', 0.15, 'volatility of volatility (Default 15%)')
FLAGS = flags.FLAGS

def main(argv):
    class ModelParameters:
        """
        Encapsulates model parameters
        """
        def __init__(self,
                     S0, num_period, dt, sigma, gbm_mu = 0.0,
                     jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0,
                     cir_a=0.0, r=0.0, cir_rho=0.0,
                     heston_kappa=0.0, heston_theta=0.0, heston_vol0=0.0):
            # This is the starting stock price
            self.S0 = S0
            # This is total number of simulations
            self.num_period = num_period
            # This is the delta, the rate of time e.g. 1/252 = daily, 1/12 = monthly
            self.dt = dt
            # This is the volatility of the stochastic processes
            self.sigma = sigma
            # This is the annual drift factor for geometric brownian motion
            self.gbm_mu = gbm_mu
            # This is the probability of a jump happening at each point in time
            self.lamda = jumps_lamda
            # This is the volatility of the jump size
            self.jumps_sigma = jumps_sigma
            # This is the average jump size
            self.jumps_mu = jumps_mu
            # This is the rate of mean reversion for Cox Ingersoll Ross
            self.cir_a = cir_a
            # This is the interest rate value
            self.r = r
            # This is the correlation between the wiener processes of the Heston model
            self.cir_rho = cir_rho
            # This is the rate of mean reversion for volatility in the Heston model
            self.heston_kappa = heston_kappa
            # This is the long run average variance for the Heston model
            self.heston_theta = heston_theta
            # This is the starting variance value for the Heston model
            self.heston_vol0 = heston_vol0
    
    def plot_stochastic_processes(processes, title):
        """
        This method plots a list of stochastic processes with a specified title
        :return: plots the graph of the two
        """
        plt.style.use(['bmh'])
        fig, ax = plt.subplots(1)
        fig.suptitle(title, fontsize=16)
        ax.set_xlabel('Time, t')
        ax.set_ylabel('Simulated Asset Price')
        x_axis = np.arange(0, len(processes[0]), 1)
        for i in range(len(processes)):
            plt.plot(x_axis, processes[i])
        plt.show()
        
    def _heston_sim(mp):
        """Simulate Heston model
        1). stock price
        2). instantaneous vol
    
        Returns:
            np.ndarray: stock price in shape (num_path, num_period)
            np.ndarray: instantaneous vol in shape (num_path, num_period)
        """
        # Correlated normal random variables
        W1, W2 = np.random.multivariate_normal([0,0], [[1, mp.cir_rho], [mp.cir_rho, 1]], (mp.num_period - 1, 1)).T
        
        # Initialize array for variance
        v = np.zeros((mp.num_period, 1)).T
        v[:, 0] = mp.heston_vol0**2
        
        # Initialize array for stock
        S = np.zeros((mp.num_period, 1)).T
        S[:, 0] = mp.S0
        
        # Compute the paths
        for i in range(1, mp.num_period):
            S[:, i] = S[:, i-1] * np.exp((mp.r - 0.5*v[:, i-1])*mp.dt \
                                        + np.sqrt(v[:, i-1])*np.sqrt(mp.dt)*W2[:, i-1])
            v[:, i] = np.abs(v[:, i-1] + mp.heston_kappa*(mp.heston_theta - v[:, i-1])*mp.dt \
                            + mp.sigma*np.sqrt(v[:, i-1])*np.sqrt(mp.dt)*W1[:, i-1])
        return S, v
    
    np.random.seed(FLAGS.set_seed)
    tf.random.set_seed(FLAGS.set_seed)
    
    tf.keras.backend.set_floatx('float64')
    
    directory = '.'
    
    # network for synthetic data
    activation = tf.tanh
    hidden_layer = [50,50]
    n_outputs = 1
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 5e-4, decay_steps = 100000, decay_rate=0.96, staircase=False, name=None)
    
    #input processing
    def process_input(X_input, X_input_):
    	r = tf.fill([tf.shape(input=X_input)[0],1], np.float64(0.00), name = 'r') # interest rate, if applicable
    
    	S = tf.slice(X_input, (0,0), (-1,1))
    	K = tf.slice(X_input, (0,1), (-1,1))
    	T = tf.slice(X_input, (0,2), (-1,1))
    
    	S_ = tf.slice(X_input_, (0,0), (-1,1))
    	T_ = tf.slice(X_input_, (0,1), (-1,1))
    	return S, K, T, S_, T_, r
    
    # define neural network architecture
    ann = tf.keras.Sequential(
        layers=[tf.keras.layers.Dense(hidden_layer[0], activation = activation, input_shape = (2,))] + \
    	[tf.keras.layers.Dense(hidden_layer[i], activation = activation) for i in range(1, len(hidden_layer))] + \
    	[tf.keras.layers.Dense(n_outputs, activation = tf.keras.activations.softplus)],
        name="ann")
    
    # define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # define loss function
    hedging_mse = tf.keras.losses.MeanSquaredError()
    
    # FINN self-supervised learning delta-hedging policy
    @tf.function
    def loss(X_input, X_input_):
      S, K, T, S_, T_, r = process_input(X_input, X_input_)
      with tf.GradientTape() as tape:
        tape.watch(S)
        X = tf.concat([S/(K*tf.exp(-r*T)), T], 1)
        X_ = tf.concat([S_/(K*tf.exp(-r*T_)), T_], 1)
    
        out = ann(X)
        out_ = ann(X_)
        if FLAGS.option_type == 'call':
            out = K*tf.where(tf.greater(T, 1e-3), out, tf.maximum(S/K - 1, 0))
            out_ = K*tf.where(tf.greater(T_, 1e-3), out_, tf.maximum(S_/K - 1, 0))
        elif FLAGS.option_type == 'put':
            out = K*tf.where(tf.greater(T, 1e-3), out, tf.maximum(1 - S/K, 0))
            out_ = K*tf.where(tf.greater(T_, 1e-3), out_, tf.maximum(1 - S_/K, 0))
      delta = tape.gradient(out, S)
      if FLAGS.option_type == 'call':
          delta = tf.maximum(delta, 0) # call delta bounds
          delta = tf.minimum(delta, 1) # call delta bounds
      elif FLAGS.option_type == 'put':
          delta = tf.maximum(delta, -1) # put delta bounds
          delta = tf.minimum(delta, 0) # put delta bounds
      return hedging_mse(delta*(S_-S), out_-out)
    
    @tf.function
    def grad(X_train, X_train_):
    	with tf.GradientTape() as tape:
    		tape.watch(ann.trainable_variables)
    		loss_value = loss(X_train, X_train_)
    	return loss_value, tape.gradient(loss_value, ann.trainable_variables)
    
    # define training ops
    @tf.function
    def training_op(X_train, X_train_):
    	loss_value, grads = grad(X_train, X_train_)
    	optimizer.apply_gradients(zip(grads, ann.trainable_variables))
    
    # Simulating heston model motion
    def stock_sim_path(S, T, sigma):
        mp = ModelParameters(S0=S, # initial Stock price
                         r=0, # interest rate
                         num_period= T * 250,# how many days
                         dt=1/250, #0.004, # daily
                         sigma=sigma, # vol of vol
                         cir_rho=-0.7, # correlation of heston S and sigma
                         heston_kappa=1.25, # Kappa, risk avaersion parameter
                         heston_theta=0.0225, # long term average variance, Theta
                         heston_vol0=0.15) # your initial volatility, not variance
        return _heston_sim(mp)[0][0], _heston_sim(mp)[1][0]
    
    def get_batch2(stock_path, vol_path, n, moneyness_range = (.5,2)): 
    		"""Constructs theoretical options based on the time series stock_path"""
    		picks = np.random.randint(0, len(stock_path)-1, n)
    		T = np.random.randint(1, 150, (n,1))
    		S = stock_path[picks]
    		S_ = stock_path[picks+1]
    		vol = vol_path[picks]
    		vol_ = vol_path[picks+1]
    		K = np.random.uniform(*moneyness_range, (n,1))*S
    		X = np.hstack([S, K, T/250, vol])
    		X_ = np.hstack([S_, (T-1)/250, vol_])
    		return X, X_
    
    #model training
    n_epochs = 250  # number of training epochs
    n_batches = 1000  # number of batches per epoch
    batch_size = 10000 # number of theoretical options in each batch
    T = 2 # number of years of training data
    days = int(250*T)
    
    stock_path, vol_path = stock_sim_path(100, T, FLAGS.volvol) #simulate stock path
    stock_path_test, vol_path_test = stock_sim_path(100, T, FLAGS.volvol) #simulate stock path for cross-validation
    stock_path = np.expand_dims(stock_path, axis = 1) # adding one dimension, for training purposes
    stock_path_test = np.expand_dims(stock_path_test, axis = 1)
    vol_path = np.expand_dims(vol_path, axis = 1) # adding one dimension, for training purposes
    vol_path_test = np.expand_dims(vol_path_test, axis = 1)
    X_test, X_test_ = get_batch2(stock_path_test, vol_path_test, batch_size) #get test-set
    
    #TRAINING
    losses = []
    count = 0
    print("START TRAINING")
    for epoch in range(n_epochs): # 1 epoch is going over all of the values in one stock path
    	for batch in range(n_batches): 
    		X_train, X_train_ = get_batch2(stock_path, vol_path, batch_size) # get batch
    		training_op(X_train, X_train_)
    	epoch_loss = loss(X_test, X_test_)
    	losses.append(epoch_loss)
    	print('Epoch:', epoch, 'Loss:', epoch_loss.numpy()) #, 'BS Loss:', bs_hedging_mse.eval({X_input: X_test, X_input_: X_test_}))
    	count += 1
    
    ann.save('heston_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))
    
    plt.figure()
    plt.semilogy(np.arange(count), losses)
    plt.legend()
    plt.title('Loss Function')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Losses per Epoch')
    plt.xlim([0, count])
    plt.savefig('heston_european_{}_{}_{}_loss.png'.format(FLAGS.option_type, FLAGS.volvol, FLAGS.set_seed))

if __name__ == '__main__':
    app.run(main)
