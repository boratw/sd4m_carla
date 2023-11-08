# Copyright (c) 2023 Taewoo Kim

import numpy as np
import tensorflow.compat.v1 as tf
import math
from networks.gaussian_distribution import GaussianDistribution
from networks.gaussian_encoder import GaussianEncoder
from networks.gaussian_policy import GaussianPolicy
from networks.mlp import MLP


EPS = 1e-5

class Skill_Learner:
    def __init__(self, state_len, action_len, goal_len,  traj_len, traj_track_len, latent_len, latent_body_len, task_num, name="",
        encoder_body_hidden_len=[256, 256], encoder_actor_hidden_len=[256, 256], decoder_body_hidden_len=[256, 256],
        decoder_action_hidden_len=[256, 256], vae_hidden_nonlinearity=tf.nn.tanh, random_batch=8,
        learner_lr=0.001, sampler_lr=0.001, sampler_diverse=0.9, sampler_disperse=0.5,
        learner_kl_reg=0.001, learner_unity=0.01, learner_diverse=0.005, learner_l2_reg=0.0001) :

        self.name = "Skill_Learner" + name
        body_len = traj_len // traj_track_len
        with tf.variable_scope(self.name): 
            self.input_latent = tf.placeholder(tf.float32, [None, latent_len], name="input_latent")
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_latent_body = tf.placeholder(tf.float32, [None, latent_body_len], name="input_latent_body")
            self.input_body_traj = tf.placeholder(tf.float32, [None, body_len, goal_len], name="input_body_traj")
            self.input_state_traj = tf.placeholder(tf.float32, [None, traj_len, state_len], name="input_state_traj")
            self.input_action_traj = tf.placeholder(tf.float32, [None, traj_len, action_len], name="input_action_traj")
            self.input_dropout = tf.placeholder(tf.float32, None, name="input_dropout")

            batch_size = tf.shape(self.input_state_traj)[0]
            body_batched = tf.reshape(self.input_body_traj, [-1, body_len * goal_len])
            state_batched = tf.reshape(self.input_state_traj, [-1, traj_len * state_len])
            action_batched = tf.reshape(self.input_action_traj, [-1, traj_len * action_len])
            input_batched = tf.concat([state_batched, action_batched], axis=1)

            sampling_latent =  tf.random.normal([batch_size, latent_body_len])
            
            with tf.variable_scope("Learner"):
                self.learner_encoder_body = GaussianEncoder("learner_encoder_body", body_len * goal_len, latent_body_len, encoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=body_batched, input_dropout=self.input_dropout)
                self.learner_encoder_agent = GaussianEncoder("learner_encoder_agent", traj_len * (state_len + action_len), latent_len - latent_body_len, encoder_actor_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=input_batched, input_dropout=self.input_dropout)
                encoded_latent_body = self.learner_encoder_body.reparameterized
                encoded_latent_agent = self.learner_encoder_agent.reparameterized
                encoded_latent_agent_batched = tf.reshape(self.learner_encoder_agent.reparameterized, [task_num, -1, latent_len - latent_body_len])

                encoded_latent_batched = tf.reshape(tf.tile(tf.concat([tf.stop_gradient(encoded_latent_body), encoded_latent_agent], axis=1), [1, traj_len]), [-1, latent_len])
                state_output_batched = tf.reshape(self.input_state_traj, [-1, state_len])
                action_output_batched = tf.reshape(action_batched, [-1, action_len])
                self.learner_decoder_body = GaussianDistribution("learner_decoder_body", latent_body_len,  body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=self.input_latent_body, input_dropout=self.input_dropout)
                self.learner_encoded_decoder_body = GaussianDistribution("learner_decoder_body", latent_body_len, body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=encoded_latent_body, input_dropout=self.input_dropout, reuse=True)
                self.learner_latent_decoder_body = GaussianDistribution("learner_decoder_body", latent_body_len, body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=sampling_latent, input_dropout=self.input_dropout, reuse=True)

                self.learner_decoder_action = GaussianDistribution("learner_decoder_action", latent_len,  action_len, decoder_action_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=self.input_latent, additional_input=True, additional_input_dim=state_len, additional_input_tensor=self.input_state, input_dropout=self.input_dropout, output_tanh=True)
                self.learner_encoded_decoder_action = GaussianDistribution("learner_decoder_action", latent_len, action_len, decoder_action_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=encoded_latent_batched, additional_input=True, additional_input_dim=state_len, additional_input_tensor=state_output_batched, input_dropout=self.input_dropout,
                    output_tanh=True, reuse=True)

                self.learner_output_discrete_body = tf.reshape(self.learner_decoder_body.mu, [-1, body_len, goal_len])
                self.learner_output_discrete_action = self.learner_decoder_action.output_discrete
                self.learner_output_stochastic_body = tf.reshape(self.learner_decoder_body.reparameterized, [-1, body_len, goal_len])
                self.learner_output_stochastic_action = self.learner_decoder_action.reparameterized


                self.learner_body_likelihood = tf.reduce_mean((self.learner_encoded_decoder_body.reparameterized - body_batched) ** 2) 
                self.learner_action_likelihood = tf.reduce_mean((self.learner_encoded_decoder_action.reparameterized - action_output_batched) ** 2)
                self.learner_body_kl_loss = tf.reduce_mean(self.learner_encoder_body.regularization_loss)
                self.learner_body_l2_loss = tf.reduce_mean(self.learner_encoder_body.l2_loss + self.learner_encoded_decoder_body.l2_loss)

                self.learner_agent_kl_loss = tf.reduce_mean(self.learner_encoder_agent.regularization_loss)
                self.learner_agent_l2_loss = tf.reduce_mean(self.learner_encoder_agent.l2_loss + self.learner_encoded_decoder_action.l2_loss)
                self.learner_agent_unity = tf.reduce_mean(tf.math.reduce_std(encoded_latent_agent_batched, axis=1) )
                self.encoded_latent_agent_mean = tf.reduce_mean(encoded_latent_agent_batched, axis=1)
                self.learner_agent_diverse = tf.reduce_mean(tf.math.reduce_std(self.encoded_latent_agent_mean, axis=0))
                self.encoded_latent_body_mean = tf.reduce_mean(encoded_latent_body, axis=0)
                self.encoded_latent_body_var = tf.math.reduce_variance(encoded_latent_body, axis=0)

                self.learner_optimizer_body = tf.train.AdamOptimizer(learner_lr)
                self.learner_optimizer_action = tf.train.AdamOptimizer(learner_lr)

                self.learner_train_body = self.learner_optimizer_body.minimize(loss = 
                    self.learner_body_likelihood + self.learner_body_kl_loss  * learner_kl_reg + self.learner_body_l2_loss  * learner_l2_reg,
                    var_list=[*self.learner_encoder_body.trainable_params, *self.learner_encoded_decoder_body.trainable_params])
                self.learner_train_action = self.learner_optimizer_action.minimize(loss = 
                    self.learner_action_likelihood + self.learner_agent_kl_loss  * learner_kl_reg + self.learner_agent_l2_loss  * learner_l2_reg
                    + self.learner_agent_unity * learner_unity - self.learner_agent_diverse * learner_diverse ,
                    var_list=[*self.learner_encoder_agent.trainable_params, *self.learner_encoded_decoder_action.trainable_params])
            
            with tf.variable_scope("Sampler"):
                self.sampler = GaussianDistribution("sampler", latent_body_len, body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,
                    input_tensor=self.input_latent_body, input_dropout=self.input_dropout)
                self.sampled_sampler = GaussianDistribution("sampler", latent_body_len, body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,reuse=True,
                    input_tensor=sampling_latent, input_dropout=self.input_dropout)
                self.sampler_output_discrete = tf.reshape(self.sampler.mu, [-1, body_len, goal_len])
                self.sampler_output_stochastic = tf.reshape(self.sampler.reparameterized, [-1, body_len, goal_len])
                random_input = tf.random.normal([batch_size * random_batch, latent_body_len])
                self.random_sampler = GaussianDistribution("sampler", latent_body_len, body_len * goal_len, decoder_body_hidden_len, hidden_nonlinearity=vae_hidden_nonlinearity,reuse=True,
                    input_tensor=random_input, input_dropout=self.input_dropout)

                self.sampler_params = tf.trainable_variables(scope=tf.get_variable_scope().name)

                decoder_body_traj = self.learner_latent_decoder_body.reparameterized
                self.sampler_likelihood = tf.reduce_mean(self.sampled_sampler.log_li(decoder_body_traj))
                sampler_sampled_output = tf.reshape(self.sampled_sampler.reparameterized, [-1, body_len, goal_len])
                self.sampler_disperse_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.reduce_sum(sampler_sampled_output, axis=1) ** 2, axis=1)))
                decoder_body_traj_batched = tf.tile(decoder_body_traj, [random_batch, 1])
                self.random_sampler_likelihood = tf.reduce_mean(self.random_sampler.log_li(decoder_body_traj_batched, clip_mu=True, sig_gradient=False))

                self.sampler_optimizer = tf.train.AdamOptimizer(sampler_lr)
                self.sampler_train = self.sampler_optimizer.minimize(loss = 
                    self.sampler_likelihood + self.sampled_sampler.l2_loss * learner_l2_reg
                    - self.random_sampler_likelihood * sampler_diverse - self.sampler_disperse_loss * sampler_disperse,
                    var_list=self.sampler_params)
                    


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}
            self.zero_latent_mean = np.zeros((task_num, latent_len - latent_body_len), np.float32)
            self.zero_latent_body_mean = np.zeros((latent_body_len,), np.float32)
            self.zero_latent_body_var = np.zeros((latent_body_len,), np.float32)


    def get_goal(self, input_latent, discrete=False):
        input_list = {self.input_latent_body : input_latent, self.input_dropout : 0.0}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.sampler_output_discrete, input_list)
        else:
            output = sess.run(self.sampler_output_stochastic, input_list)
        return output

    def get_learned_goal(self, input_latent, discrete=False):
        input_list = {self.input_latent_body : input_latent, self.input_dropout : 0.0}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.learner_output_discrete_body, input_list)
        else:
            output = sess.run(self.learner_output_stochastic_body, input_list)
        return output

    def get_action(self, input_state, input_latent, discrete=False):
        input_list = {self.input_state : input_state, self.input_latent : input_latent, self.input_dropout : 0.0}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.learner_output_discrete_action, input_list)
        else:
            output = sess.run(self.learner_output_stochastic_action, input_list)
        return output


    def optimize(self, input_state_traj, input_body_traj, input_action_traj,):
        input_list = {self.input_state_traj : input_state_traj, self.input_body_traj : input_body_traj, self.input_action_traj : input_action_traj,
                    self.input_dropout : 0.1}
        sess = tf.get_default_session()

        _, _, l1, l6, l7, l6_2, l7_2, l5, l10, l11, l12, l13, l14 = sess.run([self.learner_train_body, self.learner_train_action,
            self.learner_body_likelihood, self.learner_body_kl_loss, self.learner_body_l2_loss, self.learner_agent_kl_loss, self.learner_agent_l2_loss, 
            self.learner_action_likelihood, self.learner_agent_unity, self.learner_agent_diverse, self.encoded_latent_agent_mean,
            self.encoded_latent_body_mean, self.encoded_latent_body_var], input_list)
        _, l2, l3, l4 = sess.run([self.sampler_train, self.sampler_likelihood, self.random_sampler_likelihood, self.sampler_disperse_loss], input_list)

        self.log_learner_li_body += l1
        self.log_learner_li_action += l5
        self.log_learner_reg_body += l6 + l7
        self.log_learner_reg_agent += l6_2 + l7_2
        self.log_learner_unity_agent += l10
        self.log_learner_div_agent += l11
        self.log_learner_lat_mean += l12
        self.log_learner_lat_body_mean += l13
        self.log_learner_lat_body_var += l14
        self.log_sampler_li += l2
        self.log_sampler_random_li += l3
        self.log_sampler_dis += l4
        self.log_num_learner += 1
        self.log_num_learner_action += 1

        return l5

    def network_initialize(self):
        sess = tf.get_default_session()
        self.log_learner_li_body = 0
        self.log_learner_li_action = 0
        self.log_learner_reg_body = 0
        self.log_learner_reg_agent = 0
        self.log_learner_unity_agent = 0
        self.log_learner_div_agent = 0
        self.log_learner_lat_body_mean = self.zero_latent_body_mean.copy()
        self.log_learner_lat_body_var = self.zero_latent_body_var.copy()
        self.log_learner_lat_mean = self.zero_latent_mean.copy()
        self.log_sampler_li = 0
        self.log_sampler_random_li = 0
        self.log_sampler_dis = 0
        self.log_num_learner = 0
        self.log_num_learner_action = 0

    def network_update(self):
        sess = tf.get_default_session()
        self.log_learner_li_body = 0
        self.log_learner_li_action = 0
        self.log_learner_reg_body = 0
        self.log_learner_reg_agent = 0
        self.log_learner_unity_agent = 0
        self.log_learner_div_agent = 0
        self.log_learner_lat_mean = self.zero_latent_mean.copy()
        self.log_learner_lat_body_mean = self.zero_latent_body_mean.copy()
        self.log_learner_lat_body_var = self.zero_latent_body_var.copy()
        self.log_sampler_li = 0
        self.log_sampler_random_li = 0
        self.log_sampler_dis = 0
        self.log_num_learner = 0
        self.log_num_learner_action = 0

    def log_caption(self):
        return "\t" + self.name + "_Learner_Likelihood_Body\t" + self.name + "_Learner_Likelihood_Action\t" \
            + self.name + "_Learner_Body_Regularization\t" + self.name + "_Learner_Agent_Regularization\t" \
            + self.name + "_Learner_Agent_Unity\t" + self.name + "_Learner_Agent_Diverse\t" + self.name + "_Learner_Latent_Mean" \
             + self.name + "_Learner_Latent_Body_Mean" + self.name + "_Learner_Latent_Body_Var" \
            + self.name + "_Sampler_Likelihood\t" + self.name + "_Sampler_Random_Likelihood\t" + self.name + "_Sampler_Disperse\t" 
            
    
    def current_log(self):
        log_num_learner = self.log_num_learner if self.log_num_learner > 0 else 1
        log_num_learner_action = self.log_num_learner_action if self.log_num_learner_action > 0 else 1
        return "\t" + str(self.log_learner_li_body / log_num_learner) + "\t" + str(self.log_learner_li_action / log_num_learner_action) \
            + "\t" + str(self.log_learner_reg_body / log_num_learner) + "\t" + str(self.log_learner_reg_agent / log_num_learner) \
            + "\t" + str(self.log_learner_unity_agent / log_num_learner_action) + "\t" + str(self.log_learner_div_agent / log_num_learner_action)\
            + "\t" + str(self.log_learner_lat_mean / log_num_learner_action).replace("\n"," ")\
            + "\t" + str(self.log_learner_lat_body_mean / log_num_learner_action).replace("\n"," ")\
            + "\t" + str(self.log_learner_lat_body_var / log_num_learner_action).replace("\n"," ")\
            + "\t" + str(self.log_sampler_li / log_num_learner) + "\t" + str(self.log_sampler_random_li / log_num_learner) + "\t" + str(self.log_sampler_dis / log_num_learner) 

    def log_print(self):
        log_num_learner = self.log_num_learner if self.log_num_learner > 0 else 1
        log_num_learner_action = self.log_num_learner_action if self.log_num_learner_action > 0 else 1
        print ( self.name + "\n" \
            + "\tLearner_Likelihood_Body          : " + str(self.log_learner_li_body / log_num_learner) + "\n" \
            + "\tLearner_Likelihood_Action        : " + str(self.log_learner_li_action / log_num_learner_action) + "\n" \
            + "\tLearner_Body_Regularization      : " + str(self.log_learner_reg_body / log_num_learner) + "\n" \
            + "\tLearner_Agent_Regularization     : " + str(self.log_learner_reg_agent / log_num_learner_action) + "\n" \
            + "\tLearner_Agent_Unity              : " + str(self.log_learner_unity_agent / log_num_learner_action) + "\n" \
            + "\tLearner_Agent_Diverse            : " + str(self.log_learner_div_agent / log_num_learner_action) + "\n" \
            + "\tLearner_Latent_Mean              : " + str(self.log_learner_lat_mean / log_num_learner_action).replace("\n"," ") + "\n" \
            + "\tLearner_Latent_Body_Mean         : " + str(self.log_learner_lat_body_mean / log_num_learner_action).replace("\n"," ") + "\n" \
            + "\tLearner_Latent_Body_Var          : " + str(self.log_learner_lat_body_var / log_num_learner_action).replace("\n"," ") + "\n" \
            + "\tSampler_Likelihood               : " + str(self.log_sampler_li / log_num_learner) + "\n" \
            + "\tSampler_Random_Likelihood        : " + str(self.log_sampler_random_li / log_num_learner) + "\n" \
            + "\tSampler_Disperse                 : " + str(self.log_sampler_dis / log_num_learner) + "\n" )