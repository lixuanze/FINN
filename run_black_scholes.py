import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from absl import app
from absl import flags

flags.DEFINE_string('option_type', 'call', 'European Call / Put / Binary (High/Low) Option (Default European Call)')
flags.DEFINE_integer('set_seed', 413, 'training and testing seed (Default 413)')
flags.DEFINE_float('alpha', 0.07, 'real-world stock return (Default 7.0%)')
flags.DEFINE_float('volatility', 0.15, 'stock volatility (Default 15%)')
flags.DEFINE_float('binary_payoff', 1, 'binary option payoff (Default $1)')
FLAGS = flags.FLAGS

def main(argv):
    np.random.seed(FLAGS.set_seed)
    tf.random.set_seed(FLAGS.set_seed)
    tf.keras.backend.set_floatx('float64')
    
    directory = '.'
    
    # network for synthetic data
    activation = tf.tanh
    hidden_layer = [50, 50]
    n_outputs = 1
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 5e-4, decay_steps = 100000, decay_rate=0.96, staircase=False, name="lr_scheduler")
    
    #input processing
    def process_input(X_input, X_input_):
    	r = tf.fill([tf.shape(input=X_input)[0],1], np.float64(0.00), name = 'r')
    
    	S = tf.slice(X_input, (0,0), (-1,1))
    	K = tf.slice(X_input, (0,1), (-1,1))
    	T = tf.slice(X_input, (0,2), (-1,1))
    
    	S_ = tf.slice(X_input_, (0,0), (-1,1))
    	T_ = tf.slice(X_input_, (0,1), (-1,1))
    	return S, K, T, S_, T_, r
    
    # neural network architecture
    ann = tf.keras.Sequential(
        layers=[tf.keras.layers.Dense(hidden_layer[0], activation = activation, input_shape = (2,))] + \
    	[tf.keras.layers.Dense(hidden_layer[i], activation = activation) for i in range(1, len(hidden_layer))] + \
    	[tf.keras.layers.Dense(n_outputs, activation = tf.keras.activations.softplus)],
        name="ann")
    
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # loss function
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
        elif FLAGS.option_type == 'binary_call':
            out = K*tf.where(tf.greater(T, 1e-3), out, tf.where(tf.greater(S, K), FLAGS.binary_payoff/K, 0))
            out_ = K*tf.where(tf.greater(T_, 1e-3), out_, tf.where(tf.greater(S_, K), FLAGS.binary_payoff/K, 0))
        elif FLAGS.option_type == 'binary_put':
            out = K*tf.where(tf.greater(T, 1e-3), out, tf.where(tf.greater(K, S), FLAGS.binary_payoff/K, 0))
            out_ = K*tf.where(tf.greater(T_, 1e-3), out_, tf.where(tf.greater(K, S_), FLAGS.binary_payoff/K, 0))
      delta = tape.gradient(out, S)
      if FLAGS.option_type == 'call' or FLAGS.option_type == 'binary_call':
          delta = tf.maximum(delta, 0) # call delta bounds
          delta = tf.minimum(delta, 1) # call delta bounds
      elif FLAGS.option_type == 'put' or FLAGS.option_type == 'binary_put':
          delta = tf.maximum(delta, -1) # put delta bounds
          delta = tf.minimum(delta, 0) # put delta bounds
      return hedging_mse(delta*(S_-S), out_-out)
    
    #evaluate loss function and gradient
    @tf.function
    def grad(X_train, X_train_):
        with tf.GradientTape() as tape:
            loss_value = loss(X_train, X_train_)
        return loss_value, tape.gradient(loss_value, ann.trainable_variables)

    # define training ops
    @tf.function
    def training_op(X_train, X_train_):
        loss_value, grads = grad(X_train, X_train_)
        optimizer.apply_gradients(zip(grads, ann.trainable_variables))
    
    # Simulating geometric Brownian motion
    def stock_sim_path(S, alpha, delta, sigma, T, N, n):
        """Simulates geometric Brownian motion."""
        h = T/n
        mean = (alpha - delta - .5*sigma**2)*h
        vol = sigma * h**.5
        return S*np.exp((mean + vol*np.random.randn(n,N)).cumsum(axis = 0))
    
    def get_batch2(stock_path,n, moneyness_range = (.5,2)):
        """Constructs theoretical options based on the time series stock_path"""
        picks = np.random.randint(0, len(stock_path)-1, n)
        T = np.random.randint(1, 150, (n,1))
        S = stock_path[picks]
        S_ = stock_path[picks+1]
        K = np.random.uniform(*moneyness_range, (n,1))*S
        X = np.hstack([S, K, T/250])
        X_ = np.hstack([S_, (T-1)/250])
        return X, X_
    
    #model training
    n_epochs = 250  # number of training epochs
    n_batches = 1000  # number of batches per epoch
    batch_size = 10000 # number of theoretical options in each batch
    T = 2 # number of years of training data
    days = int(250*T)
    
    stock_path = stock_sim_path(100, FLAGS.alpha, 0, FLAGS.volatility, T, 1, days) #simulate stock path
    stock_path_test = stock_sim_path(100, FLAGS.alpha, 0, FLAGS.volatility, T, 1, days) #simulate stock path for cross-validation
    
    X_test, X_test_ = get_batch2(stock_path_test, batch_size)
    
    #TRAINING
    losses = []
    count = 0
    print("START TRAINING")
    for epoch in range(n_epochs):
    	for batch in range(n_batches):
    		X_train, X_train_ = get_batch2(stock_path, batch_size) # get batch of theoretical options
    		training_op(X_train, X_train_)
    	epoch_loss = loss(X_test, X_test_)
    	losses.append(epoch_loss)
    	print('Epoch:', epoch, 'Loss:', epoch_loss.numpy())
    	count += 1
    
    ann.save('gbm_european_{}_{}_{}.h5'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))
    
    plt.figure()
    plt.semilogy(np.arange(count), losses)
    plt.legend()
    plt.title('Loss Function')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Losses per Epoch')
    plt.xlim([0, count])
    plt.savefig('gbm_european_{}_{}_{}_loss.png'.format(FLAGS.option_type, FLAGS.volatility, FLAGS.set_seed))

if __name__ == '__main__':
    app.run(main)