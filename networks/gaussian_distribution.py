# Copyright (c) 2023 Taewoo Kim

import numpy as np
import tensorflow.compat.v1 as tf
import math

EPS = 1e-6


class GaussianDistribution:
    def __init__(self, name, input_dim, output_dim, hidden_sizes, hidden_nonlinearity=tf.nn.relu, reuse=False,
        input_tensor=None, additional_input=False, additional_input_dim=0, additional_input_tensor=None, random_batch=None, 
        traj_dim=None, sig_clip_min=-10.0,sig_clip_max=2.0, input_dropout=None, output_tanh=False ):
        self.random_batch = random_batch
        self.output_tanh = output_tanh
        self.output_len = output_dim
        with tf.variable_scope(name, reuse=reuse):

            if input_tensor is None:
                self.layer_input = tf.placeholder(tf.float32, [None, input_dim])
            else:
                self.layer_input = input_tensor

            if additional_input :
                if random_batch is None:
                    if additional_input_tensor is None:
                        self.layer_additional = tf.placeholder(tf.float32, [None, additional_input_dim])
                    else:
                        self.layer_additional = additional_input_tensor
                else:
                    self.layer_input = tf.tile(self.layer_input, [random_batch, 1])
                    random_tensor = tf.random.normal([tf.shape(self.layer_input)[0] // traj_dim, additional_input_dim])
                    self.layer_additional = tf.reshape(tf.tile(random_tensor, [1, traj_dim]), [-1, additional_input_dim])
            
            w1 = tf.get_variable("w1", shape=[input_dim, hidden_sizes[0]], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (input_dim + hidden_sizes[0])), math.sqrt(6.0 / (input_dim + hidden_sizes[0])), dtype=tf.float32),
                trainable=True)
            b1 = tf.get_variable("b1", shape=[hidden_sizes[0]], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc1 = tf.matmul(self.layer_input, w1) + b1
            if input_dropout is not None:
                fc1 = tf.nn.dropout(fc1, rate=input_dropout)
            if hidden_nonlinearity == tf.nn.leaky_relu:
                fc1 = tf.nn.leaky_relu(fc1, alpha=0.05)
            elif hidden_nonlinearity is not None:
                fc1 = hidden_nonlinearity(fc1)
            next_hidden_size = hidden_sizes[0]

            if additional_input :
                
                w1_add = tf.get_variable("w1_add", shape=[additional_input_dim, hidden_sizes[0]], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (additional_input_dim + hidden_sizes[0])), math.sqrt(6.0 / (additional_input_dim + hidden_sizes[0])), dtype=tf.float32),
                    trainable=True)
                b1_add = tf.get_variable("b1_add", shape=[hidden_sizes[0]], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                

                fc1 = tf.concat([fc1, tf.matmul(self.layer_additional, w1_add) + b1_add], axis=1)
                next_hidden_size = hidden_sizes[0] * 2

            w2 = tf.get_variable("w2", shape=[next_hidden_size, hidden_sizes[1]], dtype=tf.float32, 
                initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (next_hidden_size + hidden_sizes[1])), math.sqrt(6.0 / (next_hidden_size + hidden_sizes[1])), dtype=tf.float32),
                trainable=True)
            b2 = tf.get_variable("b2", shape=[hidden_sizes[1]], dtype=tf.float32, 
                initializer=tf.zeros_initializer(dtype=tf.float32),
                trainable=True)

            fc2 = tf.matmul(fc1, w2) + b2
            if input_dropout is not None:
                fc2 = tf.nn.dropout(fc2, rate=input_dropout)
            if hidden_nonlinearity == tf.nn.leaky_relu:
                fc2 = tf.nn.leaky_relu(fc2, alpha=0.05)
            elif hidden_nonlinearity is not None:
                fc2 = hidden_nonlinearity(fc2)
            if np.shape(hidden_sizes)[0] == 2:

                w3 = tf.get_variable("w3", shape=[hidden_sizes[1], output_dim * 2], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes[1] + output_dim * 2)), math.sqrt(6.0 / (hidden_sizes[1] + output_dim * 2)), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[output_dim * 2], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc3 = tf.matmul(fc2, w3) + b3
                self.layer_output = fc3
            elif np.shape(hidden_sizes)[0] == 3:
                w3 = tf.get_variable("w3", shape=[hidden_sizes[1], hidden_sizes[2]], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes[1] + hidden_sizes[2])), math.sqrt(6.0 / (hidden_sizes[1] + hidden_sizes[2])), dtype=tf.float32),
                    trainable=True)
                b3 = tf.get_variable("b3", shape=[hidden_sizes[2]], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)
                fc3 = tf.matmul(fc2, w3) + b3
                if input_dropout is not None:
                    fc3 = tf.nn.dropout(fc3, rate=input_dropout)
                if hidden_nonlinearity == tf.nn.leaky_relu:
                    fc3 = tf.nn.leaky_relu(fc3, alpha=0.05)
                elif hidden_nonlinearity is not None:
                    fc3 = hidden_nonlinearity(fc3)

                w4 = tf.get_variable("w4", shape=[hidden_sizes[2], output_dim * 2], dtype=tf.float32, 
                    initializer=tf.random_uniform_initializer(-math.sqrt(6.0 / (hidden_sizes[2] + output_dim * 2)), math.sqrt(6.0 / (hidden_sizes[2] + output_dim * 2)), dtype=tf.float32),
                    trainable=True)
                b4 = tf.get_variable("b4", shape=[output_dim * 2], dtype=tf.float32, 
                    initializer=tf.zeros_initializer(dtype=tf.float32),
                    trainable=True)

                fc4 = tf.matmul(fc3, w4) + b4
                self.layer_output = fc4

            self.mu, self.logsig = tf.split(self.layer_output, [output_dim, output_dim], 1)
            self.logsig  = tf.clip_by_value(self.logsig, -5.0, 2.0)

            self.dist = tf.distributions.Normal(loc=self.mu, scale=tf.exp(self.logsig))
            self.prior = tf.distributions.Normal(loc=tf.zeros_like(self.mu), scale=tf.ones_like(self.logsig))
            self.x = self.dist.sample()
            if output_tanh:
                self.output_discrete = tf.nn.tanh(self.mu)
                self.reparameterized = tf.nn.tanh(self.x)
                self.log_pi = (tf.reduce_sum(self.dist.log_prob(self.x), axis=1) - self.squash_correction(self.reparameterized))
            else:
                self.output_discrete = self.mu
                self.reparameterized = self.dist.sample()
                self.log_pi = tf.reduce_sum(self.dist.log_prob(self.reparameterized), axis=1)
                


            if random_batch is not None:
                reparameterized_batched = tf.reshape(self.reparameterized, [random_batch, -1, output_dim])
                self.reparameterized_average = tf.reduce_mean(reparameterized_batched, axis=0)

            self.regularization_loss = self.dist.kl_divergence(self.prior)
            self.magnitude_loss = tf.sqrt(tf.reduce_mean(self.reparameterized ** 2, axis=1)) - 1.

            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            self.l2_loss = tf.reduce_mean([tf.reduce_mean(p ** 2) for p in self.trainable_params])

    def log_li(self, x, clip_mu=False, sig_gradient=True):
        if sig_gradient:
            logsig = self.logsig
        else:
            logsig = tf.stop_gradient(self.logsig)
        if self.random_batch == None:
            if self.output_tanh:
                return (tf.reduce_sum(self.dist.log_prob(tf.math.atanh(tf.clip_by_value(action, -0.999, 0.999))), axis=1) - self.squash_correction(action)) / self.output_len  * 2.
            else:
                if clip_mu:
                    d = tf.clip_by_value(self.mu - x, -1, 1)
                else:
                    d = self.mu - x
                log_prod = 0.5 * ((d / tf.exp(logsig))  ** 2) + logsig + np.log(np.sqrt(np.pi * 2.))
                log_prod_sum = tf.reduce_sum(log_prod, axis=1)
                return log_prod_sum
        else:
            x = tf.tile(x, [self.random_batch, 1])
            if clip_mu:
                d = tf.clip_by_value(self.mu - x, -1, 1)
            else:
                d = self.mu - x
            sig = tf.exp(logsig)
            prob = 1. / (sig * np.sqrt(np.pi * 2.)) * tf.exp(-0.5 * ((d / sig)  ** 2))
            prob = tf.math.reduce_prod(prob, axis=1)
            prob = tf.reduce_mean(tf.reshape(prob, [self.random_batch, -1]), axis=0)
            return -tf.log(prob + EPS)


        
    def squash_correction(self, actions):
        return tf.reduce_sum(tf.log(1 - actions ** 2 + EPS), axis=1)
 

    def build_add_weighted(self, source, weight):
        return [ tf.assign(target, (1 - weight) * target + weight * source) for target, source in zip(self.trainable_params, source.trainable_params)]