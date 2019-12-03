"""Generic, flexible VAE, which allows certain architectural
specifications via command line arguments, e.g. number of encoder/decoder
layers, MLP or conv layers, etc.
"""
import os
from data_utils import (generate_training_data, make_dataset_pths,
                        load_or_make_dataset, make_memnet_checkpoint_dir,
                        TASKS, RECURRENT, data_recur2conv)
import tensorflow as tf
import numpy as np
import random
from path import Path

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


def restore_vars_from_list(var_list, checkpoint_dir):
    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=var_list)
        if tf.train.latest_checkpoint(checkpoint_dir) is None:
            param_vals = [None] * len(var_list)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            param_vals = sess.run(var_list)
    return param_vals


def init_vars_from_list(var, vals):
    return [v.assign(val) for v, val in zip(var, vals)]


class VAE():
    def __init__(self,
                 hidden_size,
                 latent_size,
                 activation,
                 checkpoint_dir,
                 trainingset_dir,
                 decision_size=100,
                 encoder_layers=1,
                 decoder_layers=1,
                 decision_layers=1,
                 kernel_size=3,
                 load_decision_weights=False,
                 load_memnet_weights=False,
                 task_weights=None,
                 w_reconstruction=[1., 1.],
                 w_decision=[1., 1.],
                 w_reg=1e-10,
                 w_rate=0,
                 beta0=None,
                 beta_steps=200000,
                 sampling_off=False,
                 encode_probe=False,
                 dataset='rectangle',
                 image_width=100,
                 RGB=False,
                 layer_type="MLP",
                 decision_target="same_different",
                 loss_func_dec="squared_error",
                 loss_func_recon=None,
                 dataset_size=int(1e4),
                 regenerate_steps=None,
                 input_std=10,
                 input_mean=50,
                 input_distribution_dim=-1,
                 probe_noise_std=0.1,
                 alpha=1.,
                 sample_distribution="gaussian",
                 decision_dim=1,
                 seqlen=6,
                 image_channels=3,
                 batch_size=20,
                 dropout_prob=1.0):
        if load_memnet_weights and not load_decision_weights:
            raise Exception
        if "float" in dataset and layer_type == "conv":
            raise NotImplementedError
        if dataset == "letter" and len(task_weights) != 4:
            raise Exception
        if layer_type == "conv" and hidden_size % 2:
            raise Exception("Kernel size must be odd number")
        self.name = "VAE"
        self.load_memnet_weights = load_memnet_weights
        activation_dict = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
            "elu": tf.nn.elu
        }
        self.dataset = dataset
        if "addednoiseprobe" in dataset:
            # Additive noise standard deviation when using
            # 'pink_addednoiseprobe' or 'bandpass_addednoiseprobe'
            self.probe_noise_std = probe_noise_std
        else:
            self.probe_noise_std = ""
        self.dataset_size = int(dataset_size)
        self.regenerate_steps = regenerate_steps
        self.batch_size = batch_size
        self.input_mean = input_mean
        self.input_std = input_std
        self.input_distribution_dim = input_distribution_dim
        self.alpha = alpha
        self.stimulus_dims = 2  # TODO: change for other datasets besides plants
        if task_weights is None:
            if decision_dim is None:
                raise Exception(
                    "Need to specify either task_weights or decision_dim")
            else:
                task_weights = [1] * decision_dim
        else:
            if hasattr(task_weights, '__len__'):
                if decision_dim is None:
                    decision_dim = len(task_weights)
                elif not decision_dim == len(task_weights):
                    raise Exception(
                        "decision_dim and task_weights mismatched for length")
            else:
                raise Exception("task_weights must be list or tuple")
        self.task_weights = np.array(task_weights)
        # if (dataset == "plants" and len(task_weights) != 1
        #         and decision_target == "same_different"):
        #     raise Exception("Decision target 'same_different' requires "
        #                     "output of length one")
        self.decision_dim = decision_dim
        self.layer_type = layer_type
        self.loss_func_dec = loss_func_dec
        self.decision_target = decision_target
        self.logistic_decision = (False if self.decision_target == "tp_dist"
                                  else True)
        if dataset in ["gabor_array", "plants_setsize"]:
            # Hardcode for now to avoid errors
            self.decision_dim = 6
            self.decision_target = "recall"
        for i in range(1, 7):
            # Hardcode for now to avoid errors
            if dataset in ["gabor_array{}".format(i),
                           "plants_setsize{}".format(i)]:
                self.decision_dim = i
                self.decision_target = "recall"
        if loss_func_recon is None:
            if dataset in TASKS["xentropy"]:
                self.loss_func_recon = "xentropy"
            else:
                self.loss_func_recon = "squared_error"
        else:
            self.loss_func_recon = loss_func_recon
        if dataset == "episodic_shape_race":
            self.decision_dim = 10
        elif dataset == "episodic_setsize":
            self.decision_dim = 12
        # Activation function
        self.activation = activation_dict[activation]
        self.image_width = image_width
        self.RGB = RGB
        if self.RGB:
            self.image_channels = 3
        else:
            self.image_channels = 1
        self.seqlen = seqlen if dataset in RECURRENT else 1
        # Size of hidden layer(s)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size  # Used if layers are convolutional
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.decision_layers = decision_layers
        self.latent_size = latent_size
        self.decision_size = decision_size
        self.w_rate = float(
            w_rate)  # Initialize to this value, but possibly change
        self.beta1 = w_rate  # Final value for rate_loss_weight
        self.beta0 = beta0  # Initial value for rate_loss_weight
        if self.beta0 is not None and self.beta0 > self.beta1:
            raise Exception("beta1 must be greater than beta0")
        self.beta_steps = beta_steps
        self.w_reconstruction_enc, self.w_reconstruction_dec = w_reconstruction
        self.w_reconstruction_enc = float(self.w_reconstruction_enc)
        self.w_reconstruction_dec = float(self.w_reconstruction_dec)
        self.w_decision_enc, self.w_decision_dec = w_decision
        self.w_decision_enc = float(self.w_decision_enc)
        self.w_decision_dec = float(self.w_decision_dec)
        self.w_reg = float(w_reg)
        self.dropout_prob = dropout_prob
        self.sampling_off = sampling_off
        self.sample_distribution = sample_distribution
        self.encode_probe = encode_probe
        self.checkpoint_top_dir = Path(checkpoint_dir)
        self.trainingset_dir = Path(trainingset_dir)
        self.checkpoint_dir = make_memnet_checkpoint_dir(checkpoint_dir, self)
        # if load_decision_weights:
        #     # Maybe deprecated...
        #     make_memnet_checkpoint_dir(
        #         checkpoint_dir, dataset, image_width, 0, 1, 1, hidden_size,
        #         latent_size, decision_size, encoder_layers, decoder_layers,
        #         [1, 1, 1], layer_type, input_mean, input_std,
        #         input_distribution_dim)
        # else:
        #     self.decision_checkpoint_dir = None
        self.decision_checkpoint_dir = None
        print("Using {} activation".format(activation))
        print("Using {} loss function for reconstruction error".format(
            self.loss_func_recon))
        print("Loss weights (rate, reconstruction, decision): ",
              (self.w_rate, w_reconstruction, w_decision))
        self.sess = None

    def _make_layer(self,
                    inputs,
                    ltype="MLP",
                    activation=None,
                    name=None,
                    kernel_regularizer=None,
                    bias_regularizer=None,
                    transpose=False,
                    reuse=False,
                    filters=None,
                    size=None):
        if ltype == "MLP":
            if size is None:
                size = self.hidden_size
            layer = tf.layers.dense(
                inputs,
                size,
                activation=activation,
                name=name,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                reuse=reuse)
        elif ltype == "conv":
            if transpose:
                layerfunc = tf.layers.conv2d_transpose
            else:
                layerfunc = tf.layers.conv2d
            if filters is None:
                filters = self.hidden_size
            layer = layerfunc(
                inputs,
                filters,
                self.kernel_size,
                activation=activation,
                name=name,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                reuse=reuse)
        layer = tf.nn.dropout(layer, self.dropout_prob)
        return layer

    def _make_placeholders(self):
        if self.layer_type == "MLP":
            self.inputs = tf.placeholder(
                tf.float32, [None, self.in_len], name='inputs')
            self.targets = tf.placeholder(
                tf.float32, [None, self.out_len], name='targets')
            self.probes = tf.placeholder(
                tf.float32, [None, self.out_len], name='probes')
        elif self.layer_type == "conv":
            self.inputs = tf.placeholder(
                tf.float32, [
                    None, self.image_width, self.image_width,
                    self.image_channels * self.seqlen
                ],
                name='inputs')
            self.targets = tf.placeholder(
                tf.float32, [
                    None, self.image_width, self.image_width,
                    self.image_channels * self.seqlen
                ],
                name='targets')
            self.probes = tf.placeholder(
                tf.float32, [
                    None, self.image_width, self.image_width,
                    self.image_channels * self.seqlen
                ],
                name='probes')
        self.true_change_prob = tf.placeholder(
            tf.float32, [None, 1], name="true_change_prob")
        self.recall_targets = tf.placeholder(
            tf.float32, [None, self.decision_dim], name="recall_targets")

    def _do_sample(self):
        if self.sampling_off:
            self.latent = self.latent_mu
        else:
            if self.sample_distribution == "uniform":
                self.latent = self.latent_mu + tf.random_uniform(
                    tf.shape(self.latent_mu), minval=-0.5, maxval=0.5)
            elif self.sample_distribution == "gaussian":
                self.eps = tf.random_normal(
                    tf.shape(self.latent_mu), name='eps')
                self.latent = self.latent_mu + self.eps * tf.exp(
                    self.latent_logsigma)
            else:
                raise NotImplementedError

    def _encoder_output(self, lyr, reuse=False):
        if self.layer_type == "MLP":
            encoder_out = tf.layers.flatten(lyr)
        elif self.layer_type == "conv":
            encoder_out = tf.layers.conv2d(
                lyr,
                self.seqlen,
                1,
                activation=None,
                name="encoder_out",
                reuse=reuse)
            encoder_out = tf.layers.flatten(encoder_out)
        else:
            raise NotImplementedError
        return encoder_out

    def build(self):
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.w_reg)
        self.global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[
                tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP
            ])
        if self.dataset in TASKS["images"]:
            # Length of output vector
            self.out_len = self.image_width**2 * self.image_channels
            self.in_len = self.out_len  # Length of an input vector
        elif self.dataset == "2Dfloat":
            self.in_len = 2
            self.out_len = 2
        elif self.dataset == "NDim_float":
            self.in_len = self.image_width
            self.out_len = self.image_width
        else:
            raise NotImplementedError
        self._make_placeholders()

        # Encoder
        self.hidden0_list = [
            self._make_layer(
                self.inputs,
                ltype=self.layer_type,
                activation=self.activation,
                name="hidden0_0",
                kernel_regularizer=self.l2_reg,
                bias_regularizer=self.l2_reg)
        ]
        for i in range(self.encoder_layers - 1):
            self.hidden0_list.append(
                self._make_layer(
                    self.hidden0_list[-1],
                    ltype=self.layer_type,
                    activation=self.activation,
                    name="hidden0_" + str(i + 1),
                    kernel_regularizer=self.l2_reg,
                    bias_regularizer=self.l2_reg))
        # Latent layer
        self.encoder_out = self._encoder_output(self.hidden0_list[-1])
        self.latent_mu = tf.layers.dense(
            self.encoder_out,
            self.latent_size,
            activation=None,
            name="latent_mu",
            kernel_regularizer=self.l2_reg,
            bias_regularizer=self.l2_reg)
        self.latent_logsigma = tf.layers.dense(
            self.encoder_out,
            self.latent_size,
            activation=None,
            name="latent_logsigma",
            kernel_regularizer=self.l2_reg,
            bias_regularizer=self.l2_reg)
        # Sample new latent values
        self._do_sample()
        # Decoder
        if self.layer_type == "conv":
            # Calculate width of first decoder conv layer (should result in
            # final decoder output with true image size)
            h_size = (self.image_width -
                      (self.kernel_size - 1) * self.decoder_layers)
            self.hidden1_list = [
                tf.reshape(
                    tf.layers.dense(
                        self.latent,
                        h_size**2 * self.seqlen,
                        activation=self.activation,
                        name="hidden1_0",
                        kernel_regularizer=self.l2_reg,
                        bias_regularizer=self.l2_reg),
                    [-1, h_size, h_size, self.seqlen])
            ]
        else:
            self.hidden1_list = [
                tf.reshape(
                    tf.layers.dense(
                        self.latent,
                        self.hidden_size,
                        activation=self.activation,
                        name="hidden1_0",
                        kernel_regularizer=self.l2_reg,
                        bias_regularizer=self.l2_reg),
                    tf.shape(self.hidden0_list[-1]))
            ]
        for i in range(self.decoder_layers - 1):
            self.hidden1_list.append(
                self._make_layer(
                    self.hidden1_list[-1],
                    ltype=self.layer_type,
                    transpose=True,
                    activation=self.activation,
                    name="hidden1_" + str(i + 1),
                    kernel_regularizer=self.l2_reg,
                    bias_regularizer=self.l2_reg))
        self.reconstruction = self._make_layer(
            self.hidden1_list[-1],
            ltype=self.layer_type,
            transpose=True,
            activation=None,
            filters=self.image_channels * self.seqlen,
            size=self.out_len,
            name="reconstruction")
        self.reconstruction_sig = tf.sigmoid(self.reconstruction)
        # Implement part of graph for comparing latent representation of probe
        # to that of target (change-detection)

        # This part of graph is only used if we want to encode the probe
        # using the same layers/weights that we used to encode the target.
        self.probes_hidden_list = [
            self._make_layer(
                self.probes,
                ltype=self.layer_type,
                activation=self.activation,
                name="hidden0_0",
                reuse=True)
        ]
        for i in range(self.encoder_layers - 1):
            self.probes_hidden_list.append(
                self._make_layer(
                    self.probes_hidden_list[-1],
                    ltype=self.layer_type,
                    activation=self.activation,
                    name="hidden0_" + str(i + 1),
                    reuse=True))
        self.probes_latent = tf.layers.dense(
            self._encoder_output(self.probes_hidden_list[-1], reuse=True),
            self.latent_size,
            activation=None,
            name="latent_mu",
            reuse=True)

        # Ground truth probability of change trial decision, determined by
        # task weights (reflects a perceptual distance in the true latent
        # space used to generate stimuli; can be set artificially or
        # correspond to subject responses)
        decision_dim = self.decision_dim
        # MLP to map from target and probe (concatenated) to probability of
        # same/different
        if self.encode_probe:
            self.decision_hidden0 = tf.layers.dense(
                tf.concat([tf.layers.flatten(self.probes_latent), self.latent],
                          1),
                self.decision_size,
                activation=self.activation,
                name="decision_hidden0")
        else:
            self.decision_hidden0 = tf.layers.dense(
                tf.concat([tf.layers.flatten(self.probes), self.latent], 1),
                self.decision_size,
                activation=self.activation,
                name="decision_hidden0",
                kernel_regularizer=self.l2_reg,
                bias_regularizer=self.l2_reg)
        self.decision_hidden_list = [self.decision_hidden0]
        for i in range(self.decision_layers - 1):
            self.decision_hidden_list.append(
                tf.layers.dense(
                    self.decision_hidden_list[-1],
                    self.decision_size,
                    activation=self.activation,
                    name="decision_hidden" + str(i + 1),
                    kernel_regularizer=self.l2_reg,
                    bias_regularizer=self.l2_reg))
        self.decision_distance = tf.layers.dense(
            self.decision_hidden_list[-1],
            decision_dim,
            activation=None,
            name="decision_distance",
            kernel_regularizer=self.l2_reg,
            bias_regularizer=self.l2_reg)
        self.decision = self.decision_distance
        self.decision_sig = tf.sigmoid(self.decision)

    def _make_scaffold(self):
        """Organize checkpoint restorations, make training scaffold.
        This code is mostly devoted to allowing us to separately load
        weights from file for different parts of the network (specifically,
        to load the decision module separately from the autoencoder module.
        """
        if self.decision_checkpoint_dir is None:
            # All vars that go in training checkpoint
            self.memnet_vars = [v for v in tf.global_variables()]
        else:
            # If preloading decision MLP weights, exclude them from saver
            self.memnet_vars = [
                v for v in tf.global_variables() if "decision" not in v.name
            ]
        self.decision_vars = [
            v for v in tf.global_variables() if "decision" in v.name
        ]
        self.saver = tf.train.Saver(self.memnet_vars, save_relative_paths=True)
        if self.decision_checkpoint_dir is not None:
            self.decision_param_vals = restore_vars_from_list(
                self.decision_vars, self.decision_checkpoint_dir)
        else:
            self.decision_param_vals = None
        if self.load_memnet_weights:
            self.memnet_param_vals = restore_vars_from_list(
                self.memnet_vars, self.decision_checkpoint_dir)
        else:
            self.memnet_param_vals = restore_vars_from_list(
                self.memnet_vars, self.checkpoint_dir)
        self.scaffold = tf.train.Scaffold(
            saver=self.saver,
            init_op=[
                var.assign(val) if 'global_step' not in var.name
                and val is not None else tf.variables_initializer([var])
                for var, val in zip(self.memnet_vars, self.memnet_param_vals)
            ],
            ready_for_local_init_op=tf.report_uninitialized_variables(
                self.memnet_vars),
            local_init_op=init_vars_from_list(self.decision_vars,
                                              self.decision_param_vals)
            if self.decision_param_vals is not None else None)

    def train(self, training_steps, report_interval, learning_rate,
              pregenerate_data):

        print("Saving checkpoints to ", self.checkpoint_dir)
        if "plant" in self.dataset:
            # Never generate plants training data on the fly
            pregenerate_data = True
        # Loss functions

        # Regularization
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.contrib.layers.apply_regularization(
            self.l2_reg, reg_variables) / self.batch_size
        self.reg_loss *= self.w_reg
        if self.loss_func_recon == "squared_error":
            self.reconstruction_loss_unweighted = tf.reduce_mean(
                tf.square(self.reconstruction -
                          self.targets))
        elif self.loss_func_recon == "absolute_value":
            self.reconstruction_loss_unweighted = tf.reduce_mean(
                tf.abs(self.reconstruction - self.targets))
        elif self.loss_func_recon == "xentropy":
            self.reconstruction_loss_unweighted = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.reconstruction,
                    labels=self.targets))
        else:
            raise NotImplementedError
        self.reconstruction_loss = self.reconstruction_loss_unweighted
        self.rate_loss_unweighted = -0.5 * (
            tf.to_float(tf.reduce_prod(tf.shape(self.latent_mu))) +
            tf.reduce_sum(2. * self.latent_logsigma) - tf.reduce_sum(
                self.latent_mu**2) - tf.reduce_sum(
                    tf.exp(2. * self.latent_logsigma))) / self.batch_size
        # Define rate loss weight relative to global step
        if self.beta0 is not None:
            self.w_rate = tf.minimum(
                self.beta1, (self.beta1 - self.beta0) * tf.to_float(
                    self.global_step) / float(self.beta_steps))
        self.rate_loss = self.rate_loss_unweighted * self.w_rate
        if self.loss_func_dec == "squared_error":
            lfunc = tf.square
        elif self.loss_func_dec == "absolute_value":
            lfunc = tf.abs
        elif self.loss_func_dec == "cubic":
            lfunc = lambda x: tf.pow(tf.abs(x), 3)
        else:
            raise NotImplementedError
        if self.decision_target == "same_different":
            # Change-probability as decision target
            self.decision_loss_unweighted = tf.reduce_mean(
                lfunc(self.true_change_prob -
                      self.decision_sig))
        elif self.decision_target == "tp_dist":
            # Target-probe perceptual distance as decision target
            self.decision_loss_unweighted = tf.reduce_mean(
                lfunc(self.true_change_prob -
                      self.decision_sig))
        elif self.decision_target == "recall":
            # Experimenter-chosen stimulus dimensions as decision target
            self.decision_loss_unweighted = tf.reduce_mean(
                lfunc(self.recall_targets - self.decision))
        else:
            raise NotImplementedError("Unknown decision target type")
        self.decision_loss = self.decision_loss_unweighted
        self.total_loss = self.reconstruction_loss_unweighted * \
            self.w_reconstruction_dec + self.decision_loss_unweighted * \
            self.w_decision_dec + self.rate_loss_unweighted * self.w_rate

        # Summary writer
        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
        tf.summary.scalar('decision_loss', self.decision_loss)
        tf.summary.scalar('rate_loss', self.rate_loss)
        self.merged_summaries = tf.summary.merge_all()
        self.summaries_writer = tf.summary.FileWriter(
            self.checkpoint_dir + '/summaries', graph=tf.get_default_graph())

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Gradients (compute manually wrt each term of loss function)
        grads_reg = tf.clip_by_global_norm(
            tf.gradients(self.reg_loss, tf.trainable_variables()), 50.)[0]
        grads_reconstruction = tf.clip_by_global_norm(
            tf.gradients(self.reconstruction_loss, tf.trainable_variables()),
            50.)[0]
        grads_decision = tf.clip_by_global_norm(
            tf.gradients(self.decision_loss, tf.trainable_variables()), 50.)[0]
        grads_rate = tf.clip_by_global_norm(
            tf.gradients(self.rate_loss, tf.trainable_variables()), 50.)[0]

        # Weight encoder and decoder differently in parameter updates
        # (This allows, for example, forcing the reconstruction
        # term to only backpropagate through the decoder, and not affect
        # the encoder parameters)
        with tf.control_dependencies([
                g for g in grads_reconstruction + grads_decision + grads_rate
                if g is not None
        ]):

            grads_zipped = []
            tvars = tf.trainable_variables()
            for i, var in enumerate(tvars):
                g_dec = grads_decision[i] if grads_decision[i] is not None \
                    else 0  # Set to zero if grad is None
                g_recon = grads_reconstruction[i] if grads_reconstruction[i] \
                    is not None else 0
                g_rate = grads_rate[i] if grads_rate[i] is not None else 0
                g_reg = grads_reg[i] if grads_reg[i] is not None else 0
                if ("hidden0" in var.name or "latent" in var.name) and \
                        "decision" not in var.name:
                    g_dec *= self.w_decision_enc
                    g_recon *= self.w_reconstruction_enc
                else:
                    g_dec *= self.w_decision_dec
                    g_recon *= self.w_reconstruction_dec
                grads_zipped.append((g_dec + g_recon + g_rate + g_reg, var))
            self.train_op = self.optimizer.apply_gradients(
                grads_zipped, global_step=self.global_step)

        # Make training scaffold
        # self._make_scaffold()
        self.saver = tf.train.Saver(save_relative_paths=True)
        # Training hooks
        self.report_interval = report_interval
        if self.report_interval > 0:
            hooks = [
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=self.checkpoint_dir,
                    save_steps=self.report_interval,
                    saver=self.saver)
            ]
        else:
            hooks = []
        # Handle dataset
        if pregenerate_data:
            # Make dataset dir and path
            dataset_dir, dataset_pth = make_dataset_pths(self)
            print("dataset path: ", dataset_pth)
            # Load or make dataset
            X, Probes, Change_var, Recall_targets = load_or_make_dataset(
                self, dataset_pth, dataset_dir, self.dataset_size)
            if self.dataset in TASKS["recurrent"]:
                # Use feedforward CNN instead of LSTM layers for recurrent
                # dataset.
                # Check that layers are conv, otherwise abort
                if not self.layer_type == "conv":
                    raise Exception(
                        "Dataset '{}' is recurrent, so layer type must be conv"
                        ".".format(self.dataset))
                # Reshape data so that time steps are stacked along same dim as
                # image channels
                # (batch, t, W, H, C) --> (batch, W, H, Cxt)
                X = data_recur2conv(X)
                Probes = data_recur2conv(Probes)

        # Begin training
        print("Beginning training.")
        with tf.train.SingularMonitoredSession(
                checkpoint_dir=self.checkpoint_dir,
                hooks=hooks,
                # scaffold=self.scaffold
                ) as sess:
            start_iteration = sess.run(self.global_step)
            for step in range(start_iteration, training_steps):
                if pregenerate_data:
                    regenerate = (
                        self.regenerate_steps > 0 and step > 0 and not
                        step % self.regenerate_steps)
                    if regenerate:
                        # Regenerate dataset periodically to avoid overfitting
                        del X, Probes, Change_var, Recall_targets
                        print("Regenerating dataset...")
                        (X, Probes, Change_var,
                         Recall_targets) = generate_training_data(
                             self.dataset_size,
                             self.image_width,
                             task_weights=self.task_weights,
                             dataset=self.dataset,
                             conv=self.layer_type == "conv",
                             data_dir=self.trainingset_dir,
                             mean=self.input_mean,
                             std=self.input_std,
                             dim=self.input_distribution_dim,
                             probe_noise_std=self.probe_noise_std,
                             logistic_decision=self.logistic_decision,
                             seqlen=self.seqlen)
                    batch_inds = np.array([
                        random.randrange(0, self.dataset_size)
                        for i in range(self.batch_size)
                    ])
                    # Select items for training batch
                    x = X[batch_inds]
                    probes = Probes[batch_inds]
                    change_var = Change_var[batch_inds] if Change_var is \
                        not None else None
                    recall_targets = Recall_targets[batch_inds] if \
                        Recall_targets is not None else None
                else:
                    # Generate data on the fly
                    (x, probes, change_var,
                     recall_targets) = generate_training_data(
                         self.batch_size,
                         self.image_width,
                         task_weights=self.task_weights,
                         dataset=self.dataset,
                         conv=self.layer_type == "conv",
                         data_dir=self.trainingset_dir,
                         mean=self.input_mean,
                         std=self.input_std,
                         dim=self.input_distribution_dim,
                         probe_noise_std=self.probe_noise_std,
                         logistic_decision=self.logistic_decision,
                         seqlen=self.seqlen)
                    if self.dataset in TASKS["recurrent"]:
                        x = data_recur2conv(x)
                        probes = data_recur2conv(probes)
                if self.decision_target == "recall":
                    # If decision target is stimulus value, no need to input
                    # a probe image
                    probes = np.zeros_like(x)
                fdict = {
                    self.inputs: x,
                    self.targets: x,
                    self.probes: probes,
                    self.true_change_prob: change_var,
                    self.recall_targets: recall_targets
                }
                sess.run(self.train_op, feed_dict=fdict)
                if step % self.report_interval == 0:
                    summary, rloss, ploss, dloss, loss = sess.run(
                        [
                            self.merged_summaries, self.rate_loss_unweighted,
                            self.reconstruction_loss_unweighted,
                            self.decision_loss_unweighted, self.total_loss
                        ],
                        feed_dict=fdict)
                    print("Step: ", step)
                    print("Losses: total: {:.5f}, pixel: {:.5f}, rate: "
                          "{:.5f}, decision: {:.5f}".format(
                              loss, ploss, rloss, dloss))
                    self.summaries_writer.add_summary(summary, step)
                    # Test overfitting
                    (x, probes, change_var,
                     recall_targets) = generate_training_data(
                         self.batch_size,
                         self.image_width,
                         task_weights=self.task_weights,
                         dataset=self.dataset,
                         conv=self.layer_type == "conv",
                         data_dir=self.trainingset_dir,
                         mean=self.input_mean,
                         std=self.input_std,
                         dim=self.input_distribution_dim,
                         probe_noise_std=self.probe_noise_std,
                         logistic_decision=self.logistic_decision,
                         seqlen=self.seqlen)
                    fdict = {
                        self.inputs: x,
                        self.targets: x,
                        self.probes: probes,
                        self.true_change_prob: change_var,
                        self.recall_targets: recall_targets
                    }
                    rloss, ploss, dloss, loss = sess.run([
                        self.rate_loss_unweighted,
                        self.reconstruction_loss_unweighted,
                        self.decision_loss_unweighted, self.total_loss
                    ],
                                                         feed_dict=fdict)
                    print(
                        "Holdout losses: total: {:.5f}, pixel: {:.5f}, rate: "
                        "{:.5f}, decision: {:.5f}".format(
                            loss, ploss, rloss, dloss))

    # def train0(self, training_steps, report_interval, lr, dataset_size, alpha=1.):
    #     # Optimizer
    #     trainable_variables = tf.trainable_variables()
    #     self.lr = lr
    #     self.optimizer = tf.train.AdamOptimizer(self.lr)
    #     self.grads, _ = tf.clip_by_global_norm(
    #         tf.gradients(self.loss, trainable_variables), 50.)
    #     # Training op
    #     self.train_op = self.optimizer.apply_gradients(
    #         zip(self.grads, trainable_variables))
    #     self.train_op = self.optimizer.minimize(self.loss,
    #                                             global_step=self.global_step)
    #     self.report_interval = report_interval
    #     self.saver = tf.train.Saver()
    #     if self.report_interval > 0:
    #             hooks = [
    #                 tf.train.CheckpointSaverHook(
    #                     checkpoint_dir=self.checkpoint_dir,
    #                     save_steps=self.report_interval,
    #                     saver=self.saver)
    #             ]
    #     else:
    #         hooks = []

    #     # Pre-generate data
    #     self.dataset_size = dataset_size
    #     self.layer_type = "MLP"  # Add these attributes to VAE for compatibility
    #     self.task_weights = [1., 1.]
    #     self.probe_noise_std = ""
    #     self.alpha = alpha
    #     dataset_dir, dataset_pth = make_dataset_pths(self)
    #     print("Dataset path: ", dataset_pth)
    #     X, _, _ = load_or_make_dataset(
    #         self, dataset_pth, dataset_dir, self.dataset_size)

    #     with tf.train.SingularMonitoredSession(
    #             checkpoint_dir=self.checkpoint_dir, hooks=hooks) as sess:
    #         start_iteration = sess.run(self.global_step)
    #         for step in range(start_iteration, training_steps):
    #             # Generate data on the fly
    #             # _, x = generate_training_data(self.batch_size, self.image_width)
    #             # x, _, _ = generate_training_data(self.batch_size, self.image_width,
    #             #                                  dataset=self.dataset)
    #             batch_inds = np.array([random.randrange(0, self.dataset_size)
    #                                    for i in range(self.batch_size)])
    #             x = X[batch_inds]
    #             # xhat = sess.run(self.outputs, feed_dict={self.inputs: x,
    #             #                                          self.targets: x})
    #             # from ipdb import set_trace as BP; BP()
    #             sess.run([self.train_op],
    #                      feed_dict={self.inputs: x, self.targets: x})
    #             if step % self.report_interval == 0:
    #                 summary, rloss, KLloss, loss = sess.run(
    #                     [self.merged_summaries, self.reconstruction_loss,
    #                      self.kl_loss, self.loss],
    #                     feed_dict={self.inputs: x, self.targets: x})
    #                 print("Step: ", step)
    #                 print ("total loss: {:.3f}, KL loss: {:.3f}, "
    #                        "reconstruction loss: {:.3f}".format(
    #                            loss, KLloss, rloss))
    #                 self.summaries_writer.add_summary(summary, step)

    def predict(self, x, keep_session=False):
        if self.dataset in TASKS["xentropy"]:
            target_var = self.reconstruction_sig
        else:
            target_var = self.reconstruction
        if keep_session:
            if self.sess is None:
                # self._make_scaffold()
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    )
            y = self.sess.run(target_var, feed_dict={self.inputs: x})
        else:
            # self._make_scaffold()
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    ) as sess:
                y = sess.run(target_var, feed_dict={self.inputs: x})
        return y

    def predict_response(self, x, y, keep_session=False, sigmoid=True):
        """Predict output of decision module
        """
        if sigmoid:
            decision_var = self.decision_sig
        else:
            decision_var = self.decision
        if keep_session:
            if self.sess is None:
                # self._make_scaffold()
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    )
            decision = self.sess.run(
                decision_var, feed_dict={
                    self.inputs: x,
                    self.probes: y
                })
        else:
            # self._make_scaffold()
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    ) as sess:
                decision = sess.run(
                    decision_var, feed_dict={
                        self.inputs: x,
                        self.probes: y
                    })
        return decision

    def predict_both(self, x, y, keep_session=False, sigmoid=True):
        if self.dataset in TASKS["xentropy"]:
            target_var = self.reconstruction_sig
        else:
            target_var = self.reconstruction
        if sigmoid:
            decision_var = self.decision_sig
        else:
            decision_var = self.decision
        if keep_session:
            if self.sess is None:
                # self._make_scaffold()
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    )
            y, decision = self.sess.run([target_var, decision_var],
                                        feed_dict={
                                            self.inputs: x,
                                            self.probes: y
                                        })
        else:
            # self._make_scaffold()
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir,
                    # scaffold=self.scaffold
                    ) as sess:
                y, decision = sess.run([target_var, decision_var],
                                       feed_dict={
                                           self.inputs: x,
                                           self.probes: y
                                       })
        return y, decision

    def encode(self, x, keep_session=False):
        if keep_session:
            if self.sess is None:
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
            z = self.sess.run(self.latent, feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                z = sess.run(self.latent, feed_dict={self.inputs: x})
        return z

    def encode_presample(self, x, keep_session=False):
        if keep_session:
            if self.sess is None:
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
            mu, sigma = self.sess.run([self.latent_mu, self.latent_logsigma],
                                      feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                mu, sigma = sess.run([self.latent_mu, self.latent_logsigma],
                                     feed_dict={self.inputs: x})
        return mu, sigma


def add_parser_args(parser):
    # Setup
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Activation for hidden units")
    parser.add_argument(
        "--trainingset_dir",
        type=str,
        default=TRAININGSET_DIR,
        help="Checkpoint directory (within which a "
        "subdirectory will be created for the specific "
        "parameters)")
    # Training
    parser.add_argument(
        "--report",
        type=int,
        default=1000,
        help="Report interval (i.e. printing progress)")
    parser.add_argument(
        "--lr_distortion",
        type=float,
        default=0.001,
        help="Learning rate for distortion loss")
    parser.add_argument(
        "--lr_rate",
        type=float,
        default=0.001,
        help="Learning rate for rate loss")
    parser.add_argument(
        "--steps",
        type=int,
        default=100000000,
        help="Number of training steps")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--pregenerate_data",
        action="store_true",
        help="Generate a fixed dataset before training"
        " to save on computation time.")
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10000,
        help="Size of dataset to pre-generate or load.")
    parser.add_argument(
        "--regenerate_steps",
        type=int,
        default=-1,
        help="How often to regenerate training set (to avoid "
        "overfitting). Ignored if <= 0.")
    # Optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer")
    parser.add_argument(
        "--rate_loss_weight",
        type=float,
        default=1.,
        help="Weight on KL loss term (analogous to information"
        " 'rate').")
    parser.add_argument(
        "--beta0",
        type=float,
        default=None,
        help="For gradually increasing the rate loss weight "
        "(i.e. 'beta'). This is the starting value of beta")
    parser.add_argument(
        "--beta_steps",
        type=int,
        default=200000,
        help="When gradually increasing the rate loss weight "
        "(i.e. 'beta'), beta is increased linearly with number"
        "of training steps. This is the step count at which "
        "beta becomes equal to 'rate_loss_weight'")
    parser.add_argument(
        "--reconstruction_loss_weights",
        type=float,
        nargs="*",
        default=[1., 1.],
        help="How much to weight reconstruction loss term in "
        "total loss, separated by encoder (first value) and "
        "decoder (second value)")
    parser.add_argument(
        "--decision_loss_weights",
        type=float,
        nargs="*",
        default=[1., 1.],
        help="How much to weight decision loss term in total "
        "loss.")
    parser.add_argument(
        "--regularizer_loss_weight",
        type=float,
        default=1e-8,
        help="How much to weight regularizer loss in total "
        "loss.")
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=1.0,
        help="Probability parameter in tf.nn.dropout")
    # Architecture
    parser.add_argument(
        "--hidden", type=int, default=100, help="Number of hidden units")
    parser.add_argument(
        "--latent", type=int, default=3, help="Number of latent units")
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="When using convolutional layers, size of "
        "kernel.")
    parser.add_argument(
        "--seqlen",
        type=int,
        default=6,
        help="Number of recurrent steps (same for both "
        "encoder and decoder")
    parser.add_argument(
        "--decision_size",
        type=int,
        default=100,
        help="Number of units in layers that implement "
        "decision")
    parser.add_argument(
        "--decision_dim",
        type=int,
        default=1,
        help="Size of decision module output. Should be set to"
        " 1 for change-detection or n for recalling "
        "stimulus values of target or changes between probe "
        "and target along those dimensions (where n is the "
        "total number of stimulus dimensions, e.g. leaf width"
        " and leaf angle. Should match dimensionality in "
        "dataset.")
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=1,
        help="Number of layers in encoder.")
    parser.add_argument(
        "--decoder_layers",
        type=int,
        default=1,
        help="Number of layers in decoder.")
    parser.add_argument(
        "--decision_layers",
        type=int,
        default=1,
        help="Number of layers in decision module.")
    parser.add_argument(
        "--layer_type",
        type=str,
        default="MLP",
        help="Type of layer to use (MLP or conv)")
    parser.add_argument(
        "--l2", type=float, default=1e-8, help="Weight for l2 regularizer.")
    parser.add_argument(
        "--activation",
        type=str,
        default='tanh',
        help="Activation for hidden units")
    parser.add_argument(
        "--loss_func_reconstruction",
        type=str,
        default="squared_error",
        help="Loss function for reconstruction loss: mean-"
        "square-error ('squared_error') or cross-entropy "
        "('xentropy').")
    parser.add_argument(
        "--loss_func_decision",
        type=str,
        default="squared_error",
        help="Loss function for decision loss: 'squared_error"
        "' or 'absolute_value'.")
    parser.add_argument(
        "--decision_target",
        type=str,
        default="same_different",
        help="Which target value to use as decision loss: "
        "change-probability ('same_different'), target-probe"
        " distance ('tp_dist'), or stimulus dimension values"
        " ('recall').")
    parser.add_argument(
        "--load_decision_weights",
        action="store_true",
        help="Load pretrained weights for decision MLP layers"
        " (according to task_weights_decision), otherwise train"
        " with rest of network.")
    parser.add_argument(
        "--encode_probe",
        action="store_true",
        help="Whether to encode probe stimulus using same "
        "encoder as target. Otherwise, just put input raw "
        "pixels directly into decision module.")
    parser.add_argument(
        "--load_memnet_weights",
        action="store_true",
        help="Load pretrained weights for memory-channel layers"
        " from same checkpoint as decision weights. Weights "
        "are still trained, but note that gradients will not "
        "pass through some layers if latent_reg_loss_weight is "
        "zero. (This is by design.)")  # DEPRECATED?
    parser.add_argument(
        "--sampling_off",
        action="store_true",
        help="Turn off sampling step in channel.")
    parser.add_argument(
        "--sample_distribution",
        type=str,
        default="gaussian",
        help="Which distribution to use for sampling step at"
        "latent layer ('uniform' or 'gaussian'). Uniform does"
        "not use the units representing log-variance, and "
        "samples just like the Balle variational autoencoder "
        "model.")
    # Task
    parser.add_argument(
        "--dataset",
        type=str,
        default='plants',
        help="Which dataset to train on (see data_utils.py)")
    parser.add_argument(
        "--task_weights",
        type=float,
        nargs="*",
        default=None,
        help="Use varies by dataset. Generally, a set of weights"
        " controlling penalties for different kinds of errors"
        "  in the decision loss.")
    parser.add_argument(
        "--dim",
        type=int,
        default=-1,
        help="(Only used with datasets which can vary input distribution"
        ") Which dimension to "
        "make stimulus distribution with respect to. 0 is "
        "width, 1 is leaf angle, and -1 is uniform on both "
        "dimensions.")
    parser.add_argument(
        "--mean",
        type=float,
        default=50.,
        help="(Only for plants dataset) Mean of stimulus "
        "distribution.")
    parser.add_argument(
        "--std",
        type=float,
        default=10.,
        help="(Only for plants dataset) Standard deviation of"
        " stimulus distribution.")
    parser.add_argument(
        "--image_width",
        type=int,
        default=30,
        help="Width in pixels of target/probe images.")
    parser.add_argument(
        "--RGB",
        action="store_true",
        help="Whether stimuli "
        "are color (3 channels) or grayscale (1 channel)")
    # Noise filters dataset (DEPRECATED)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.,
        help="In pink noise images, filter exponent (1/f^alpha)")
    parser.add_argument(
        "--probe_noise_std",
        type=float,
        default=0.2,
        help="Parameter used in creating probes for filtered "
        "noise datasets ('pink_addednoiseprobe' or 'bandpass_"
        "addednoiseprobe')")
    return parser


def make_net(args):
    net = VAE(
        args.hidden,
        args.latent,
        args.activation,
        args.checkpoint_dir,
        args.trainingset_dir,
        dataset=args.dataset,
        dataset_size=args.dataset_size,
        regenerate_steps=args.regenerate_steps,
        batch_size=args.batch,
        image_width=args.image_width,
        RGB=args.RGB,
        task_weights=args.task_weights,
        decision_size=args.decision_size,
        decision_dim=args.decision_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        decision_layers=args.decision_layers,
        load_decision_weights=args.load_decision_weights,
        load_memnet_weights=args.load_memnet_weights,
        w_reconstruction=args.reconstruction_loss_weights,
        w_rate=args.rate_loss_weight,
        beta0=args.beta0,
        beta_steps=args.beta_steps,
        w_decision=args.decision_loss_weights,
        w_reg=args.regularizer_loss_weight,
        sampling_off=args.sampling_off,
        encode_probe=args.encode_probe,
        layer_type=args.layer_type,
        decision_target=args.decision_target,
        loss_func_dec=args.loss_func_decision,
        loss_func_recon=args.loss_func_reconstruction,
        input_mean=args.mean,
        input_std=args.std,
        input_distribution_dim=args.dim,
        probe_noise_std=args.probe_noise_std,
        alpha=args.alpha,
        sample_distribution=args.sample_distribution,
        seqlen=args.seqlen,
        dropout_prob=args.dropout_prob)
    return net


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_parser_args(parser)
    args = parser.parse_args()
    vae = make_net(args)
    vae.build()
    vae.train(args.steps, args.report, args.learning_rate,
              args.pregenerate_data)
