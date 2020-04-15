"""VAE designed to handle large-scale natural image datasets (specifically,
here we train on the fruits-360 dataset)
"""
import os
from data_utils import (make_memnet_checkpoint_dir,
                        get_fruit360_filenames_and_labels)
import tensorflow as tf
import numpy as np
import random
from path import Path

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


class VAE():
    def __init__(self, activation, checkpoint_dir, trainingset_dir,
                 w_reconstruction=[1., 1.],
                 w_decision=[1., 1.], w_reg=1e-10, w_rate=0, beta0=None,
                 beta_steps=200000, sampling_off=False, dataset='fruits360',
                 input_distribution_dim=-1, batch_size=64, dropout_prob=0.5,
                 regenerate_steps=1000, buffer_size=100):
        self.name = "fruits"
        activation_dict = {"relu": tf.nn.relu, "sigmoid": tf.nn.sigmoid,
                           "tanh": tf.nn.tanh, "elu": tf.nn.elu}
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.iterator = None
        self.next_element = None
        self.batch_size = batch_size
        self.regenerate_steps = regenerate_steps
        self.decision_dim = 3  # number of classes
        self.latent_size = 1000
        # Activation function
        self.activation = activation_dict[activation]
        self.pad = 12
        self.image_width = 100 + self.pad
        self.RGB = True
        self.image_channels = 3
        # Size of hidden layer(s)
        self.w_rate = w_rate  # Initialize to this value, but possibly change
        self.beta1 = w_rate  # Final value for rate_loss_weight
        self.beta0 = beta0  # Initial value for rate_loss_weight
        if self.beta0 is not None and self.beta0 > self.beta1:
            raise Exception("beta1 must be greater than beta0")
        self.beta_steps = beta_steps
        self.w_reconstruction_enc, self.w_reconstruction_dec = w_reconstruction
        self.w_decision_enc, self.w_decision_dec = w_decision
        self.w_reg = w_reg
        self.dropout_prob = dropout_prob
        self.sampling_off = sampling_off
        self.checkpoint_top_dir = Path(checkpoint_dir)
        self.trainingset_dir = Path(trainingset_dir)
        self.checkpoint_dir = make_memnet_checkpoint_dir(
            checkpoint_dir, self)
        print("Using {} activation for memnet".format(activation))
        print("Loss weights (rate, reconstruction, decision): ", (
              self.w_rate, w_reconstruction, w_decision))
        self.sess = None

    def _parse_function(self, filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_string)
        resize_pad = tf.multinomial(tf.log([[10.] * 13]), 1)[0][0]
        image = tf.image.resize_images(
            image,
            [self.image_width + resize_pad, self.image_width + resize_pad])
        image = tf.image.random_crop(
            image, [self.image_width, self.image_width, 3])
        image = tf.image.random_flip_left_right(image)
        return image / 255., label

    def _make_dataset_iterator(self, train_test, buffer_size):
        filenames, labels = get_fruit360_filenames_and_labels(
            train_test, self.trainingset_dir)
        i_shuffle_train = np.arange(len(filenames))
        np.random.shuffle(i_shuffle_train)
        filenames = [filenames[i] for i in i_shuffle_train]
        labels = [labels[i] for i in i_shuffle_train]
        data = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # data = data.map(self._parse_function,
        #                 num_parallel_calls=self.batch_size)
        # data = data.batch(self.batch_size, drop_remainder=True)
        data = data.apply(
            tf.data.experimental.map_and_batch(
                self._parse_function,
                self.batch_size,
                num_parallel_batches=1,
                drop_remainder=True
            )
        )
        data = data.apply(tf.data.experimental.shuffle_and_repeat(buffer_size))
        # data = data.repeat()
        # data = data.shuffle(1000)
        iterator = data.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element

    def _make_placeholders(self):
        self.inputs = tf.placeholder(
            tf.float32,
            [self.batch_size, self.image_width, self.image_width,
             self.image_channels],
            name='inputs')
        self.targets = tf.placeholder(
            tf.float32,
            [self.batch_size, self.image_width, self.image_width,
             self.image_channels],
            name='targets')
        self.recall_targets = tf.placeholder(tf.int32,
                                             [self.batch_size],
                                             name="recall_targets")
        # self.recall_targets = tf.one_hot(self.recall_targets,
        #                                  self.decision_dim)

    def _do_sample(self):
        self.latent_logsigma = tf.minimum(self.latent_logsigma, 10)
        if self.sampling_off:
            self.latent = self.latent_mu
        else:
            self.eps = tf.random_normal(tf.shape(self.latent_mu),
                                        name='eps')
            self.latent = self.latent_mu + self.eps * tf.exp(
                self.latent_logsigma)

    def get_batch(self, Inputs, Targets, Recall_targets):
        batch_inds = np.array([random.randrange(0, len(Inputs))
                               for i in range(self.batch_size)])
        x = Inputs[batch_inds] if Inputs is not None else None
        y = Targets[batch_inds] if Targets is not None else None
        recall_targets = Recall_targets[batch_inds] if Recall_targets is not \
            None else None
        return x, y, recall_targets

    def build(self):
        self.l2_reg = tf.contrib.layers.l2_regularizer(self.w_reg)
        self.global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                         tf.GraphKeys.GLOBAL_STEP])
        self._make_placeholders()
        padding = "same"
        # init = tf.initializers.he_uniform()
        # init = tf.keras.initializers.random_uniform()
        init = None

        # Encoder
        self.conv1_enc = tf.layers.conv2d(self.inputs,
                                          32, 3, strides=2,
                                          name="hidden0_0",
                                          padding=padding,
                                          kernel_initializer=init)
        # self.conv1_enc = tf.nn.dropout(self.conv1_enc, self.dropout_prob)
        self.conv2_enc = tf.layers.conv2d(self.conv1_enc,
                                          64, 3, strides=2,
                                          name="hidden0_1",
                                          padding=padding,
                                          kernel_initializer=init)
        # self.conv2_enc = tf.nn.dropout(self.conv2_enc, self.dropout_prob)
        # self.maxpool1_enc = tf.layers.max_pooling2d(self.conv2_enc,
        #                                             2, 2, padding=padding)
        self.conv3_enc = tf.layers.conv2d(self.conv2_enc,
                                          64, 3, strides=2,
                                          name="hidden0_2",
                                          padding=padding,
                                          kernel_initializer=init)
        # self.conv3_enc = tf.nn.dropout(self.conv3_enc, self.dropout_prob)
        self.conv4_enc = tf.layers.conv2d(self.conv3_enc,
                                          64, 3, strides=2,
                                          name="hidden0_3",
                                          padding=padding,
                                          kernel_initializer=init)
        # self.conv4_enc = tf.nn.dropout(self.conv4_enc, self.dropout_prob)
        self.conv4_enc = tf.layers.flatten(self.conv4_enc)
        self.fc1 = tf.layers.dense(self.conv4_enc, self.latent_size,
                                   activation=self.activation,
                                   name="hidden0_4",
                                   kernel_regularizer=self.l2_reg,
                                   bias_regularizer=self.l2_reg,
                                   kernel_initializer=init)
        # Latent layer
        self.latent_mu = tf.layers.dense(self.fc1, self.latent_size,
                                         activation=None,
                                         name="latent_mu",
                                         kernel_regularizer=self.l2_reg,
                                         bias_regularizer=self.l2_reg,
                                         kernel_initializer=init)
        self.latent_logsigma = tf.layers.dense(self.fc1, self.latent_size,
                                               activation=None,
                                               name="latent_logsigma",
                                               kernel_regularizer=self.l2_reg,
                                               bias_regularizer=self.l2_reg,
                                               kernel_initializer=init)
        # Sample new latent values
        self._do_sample()
        # Decoder
        self.fc2 = tf.layers.dense(self.latent, self.latent_size,
                                   activation=self.activation,
                                   kernel_regularizer=self.l2_reg,
                                   bias_regularizer=self.l2_reg,
                                   name="hidden1_prebridge",
                                   kernel_initializer=init)
        self.bridge = tf.layers.dense(self.fc2, 7 * 7 * 64,
                                      activation=self.activation,
                                      kernel_regularizer=self.l2_reg,
                                      bias_regularizer=self.l2_reg,
                                      name="hidden1_bridge",
                                      kernel_initializer=init)
        # self.bridge = tf.nn.dropout(self.bridge, self.dropout_prob)
        self.bridge = tf.reshape(self.bridge, [-1, 7, 7, 64])
        self.conv1_dec = tf.layers.conv2d_transpose(self.bridge,
                                                    64, 3, strides=2,
                                                    name="hidden1_0",
                                                    padding=padding,
                                                    kernel_initializer=init)
        # self.conv1_dec = tf.nn.dropout(self.conv1_dec, self.dropout_prob)
        self.conv2_dec = tf.layers.conv2d_transpose(self.conv1_dec,
                                                    32, 3, strides=2,
                                                    name="hidden1_1",
                                                    padding=padding,
                                                    kernel_initializer=init)
        # self.conv2_dec = tf.nn.dropout(self.conv2_dec, self.dropout_prob)
        self.conv3_dec = tf.layers.conv2d_transpose(self.conv2_dec,
                                                    32, 3,
                                                    strides=2,
                                                    name="hidden1_2",
                                                    padding=padding,
                                                    kernel_initializer=init)
        # self.conv3_dec = tf.nn.dropout(self.conv3_dec, self.dropout_prob)
        self.conv4_dec = tf.layers.conv2d_transpose(self.conv3_dec,
                                                    self.image_channels, 3,
                                                    strides=2,
                                                    name="hidden1_3",
                                                    padding=padding,
                                                    kernel_initializer=init)
        self.reconstruction = self.conv4_dec
        # Ground truth probability of change trial decision
        # MLP to map from memory to decision variable
        # self.decision_hidden = tf.layers.dense(self.latent, 500,
        #                                        activation=self.activation,
        #                                        name="decision_hidden",
        #                                        kernel_regularizer=self.l2_reg,
        #                                        bias_regularizer=self.l2_reg,
        #                                        kernel_initializer=init)
        # self.decision_hidden = tf.nn.dropout(self.decision_hidden,
        #                                      self.dropout_prob)
        self.decision = tf.layers.dense(self.latent, self.decision_dim,
                                        activation=None, name="decision",
                                        kernel_regularizer=self.l2_reg,
                                        bias_regularizer=self.l2_reg,
                                        kernel_initializer=init)
        self.decision_sig = tf.sigmoid(self.decision)

    def train(self, training_steps, report_interval,
              learning_rate):

        print("Saving checkpoints to ", self.checkpoint_dir)

        # Loss functions

        # Regularization
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.contrib.layers.apply_regularization(
            self.l2_reg, reg_variables) / self.batch_size
        self.reconstruction_loss_unweighted = tf.reduce_mean(
            tf.square(self.reconstruction - self.targets)
        )
        self.reconstruction_loss = self.reconstruction_loss_unweighted
        self.rate_loss_unweighted = -0.5 * (
            tf.to_float(tf.reduce_prod(tf.shape(self.latent_mu))) +
            tf.reduce_sum(2. * self.latent_logsigma) -
            tf.reduce_sum(self.latent_mu ** 2) -
            tf.reduce_sum(tf.exp(2. * self.latent_logsigma))) / self.batch_size
        # Define rate loss weight relative to global step
        if self.beta0 is not None:
            self.w_rate = tf.minimum(
                self.beta1,
                (self.beta1 - self.beta0) * tf.to_float(self.global_step) /
                float(self.beta_steps))
        self.rate_loss = self.rate_loss_unweighted * self.w_rate
        # Category labels as decision targets
        self.decision_loss_unweighted = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.decision,
                labels=tf.one_hot(self.recall_targets,
                                  self.decision_dim))
        )
        self.decision_loss = self.decision_loss_unweighted
        self.total_loss = self.reconstruction_loss_unweighted * \
            self.w_reconstruction_dec + self.decision_loss_unweighted * \
            self.w_decision_dec + self.rate_loss_unweighted * self.w_rate \
            + self.reg_loss

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
        grads_reg = tf.clip_by_global_norm(tf.gradients(
            self.reg_loss,
            tf.trainable_variables()), 50.)[0]
        grads_reconstruction = tf.clip_by_global_norm(tf.gradients(
            self.reconstruction_loss,
            tf.trainable_variables()),
            50.)[0]
        grads_decision = tf.clip_by_global_norm(tf.gradients(
            self.decision_loss,
            tf.trainable_variables()), 50.)[0]
        grads_rate = tf.clip_by_global_norm(tf.gradients(
            self.rate_loss,
            tf.trainable_variables()), 50.)[0]

        # Weight encoder and decoder differently in parameter updates
        # (This allows, for example, forcing the reconstruction
        # term to only backpropagate through the decoder, and not affect
        # the encoder parameters)
        with tf.control_dependencies(
                [g for g in grads_reconstruction + grads_decision + grads_rate
                 if g is not None]):

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

        # Training hooks
        self.report_interval = report_interval
        self.saver = tf.train.Saver(save_relative_paths=True)
        if self.report_interval > 0:
                hooks = [
                    tf.train.CheckpointSaverHook(
                        checkpoint_dir=self.checkpoint_dir,
                        save_steps=self.report_interval,
                        saver=self.saver)
                ]
        else:
            hooks = []

        # Load dataset

        # Training set
        train_iterator, next_element_train = self._make_dataset_iterator(
            "train", self.buffer_size)
        # Test set
        test_iterator, next_element_test = self._make_dataset_iterator(
            "test", self.buffer_size)

        # Begin training
        print("Beginning training.")
        import time
        t0 = time.time()
        with tf.train.SingularMonitoredSession(
                checkpoint_dir=self.checkpoint_dir, hooks=hooks) as sess:
            start_iteration = sess.run(self.global_step)
            sess.run(train_iterator.initializer)
            sess.run(test_iterator.initializer)
            for step in range(start_iteration, training_steps):
                x, recall_targets = sess.run(next_element_train)
                fdict = {self.inputs: x, self.targets: x,
                         self.recall_targets: recall_targets}
                sess.run(self.train_op, feed_dict=fdict)
                if step % self.report_interval == 0:
                    print("Training time: " + str(time.time() - t0))
                    summary, rloss, ploss, dloss, loss = sess.run(
                        [self.merged_summaries, self.rate_loss_unweighted,
                         self.reconstruction_loss_unweighted,
                         self.decision_loss_unweighted, self.total_loss],
                        feed_dict=fdict)
                    print("Step: ", step)
                    print("Losses: total: {:.5f}, pixel: {:.5f}, rate: "
                          "{:.5f}, decision: {:.5f}".format(
                              loss, ploss, rloss, dloss))
                    self.summaries_writer.add_summary(summary, step)
                    x, recall_targets = sess.run(next_element_test)
                    fdict = {self.inputs: x, self.targets: x,
                             self.recall_targets: recall_targets}
                    rloss, ploss, dloss, loss = sess.run(
                        [self.rate_loss_unweighted,
                         self.reconstruction_loss_unweighted,
                         self.decision_loss_unweighted, self.total_loss],
                        feed_dict=fdict)
                    print(
                        "Holdout losses: total: {:.5f}, pixel: {:.5f}, rate: "
                        "{:.5f}, decision: {:.5f}".format(
                            loss, ploss, rloss, dloss))
                    decisions = sess.run(self.decision_sig, feed_dict=fdict)
                    decisions = np.array(
                        [np.where(d == d.max())[0][0] for d in decisions])
                    n_correct = sum(decisions == recall_targets)
                    print("Number correct, holdout: {}/{}".format(
                        str(n_correct), str(self.batch_size)))
                    t0 = time.time()

    def predict(self, x, keep_session=False):
        if keep_session:
            if self.sess is None:
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
            y = self.sess.run(self.reconstruction,
                              feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                y = sess.run(self.reconstruction,
                             feed_dict={self.inputs: x})
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
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
            decision = self.sess.run(decision_var,
                                     feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                decision = sess.run(decision_var,
                                    feed_dict={self.inputs: x})
        return decision

    def predict_both(self, x, y, keep_session=False, sigmoid=True):
        if sigmoid:
            decision_var = self.decision_sig
        else:
            decision_var = self.decision
        if keep_session:
            if self.sess is None:
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
            y, decision = self.sess.run([self.reconstruction, decision_var],
                                        feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                y, decision = sess.run([self.reconstruction, decision_var],
                                       feed_dict={self.inputs: x})
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
            mu, sigma = self.sess.run(
                [self.latent_mu, self.latent_logsigma],
                feed_dict={self.inputs: x})
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                mu, sigma = sess.run(
                    [self.latent_mu, self.latent_logsigma],
                    feed_dict={self.inputs: x})
        return mu, sigma

    def generate_batch(self, train_test, keep_session=False):
        if self.iterator is None:
            # Only make data generator once
            self.iterator, self.next_element = self._make_dataset_iterator(
                train_test, self.buffer_size)
        if keep_session:
            if self.sess is None:
                self.sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir)
                self.sess.run(self.iterator.initializer)
            x, recall_targets = self.sess.run(self.next_element)
        else:
            with tf.train.SingularMonitoredSession(
                    checkpoint_dir=self.checkpoint_dir) as sess:
                sess.run(self.iterator.initializer)
                x, recall_targets = sess.run(self.next_element)
        return x, recall_targets


def add_parser_args(parser):
    # Setup
    parser.add_argument("--checkpoint_dir", type=str,
                        default=CHECKPOINT_DIR,
                        help="Activation for hidden units")
    parser.add_argument("--trainingset_dir", type=str,
                        default=TRAININGSET_DIR,
                        help="Checkpoint directory (within which a "
                        "subdirectory will be created for the specific "
                        "parameters)")
    # Training
    parser.add_argument("--report", type=int, default=1000,
                        help="Report interval (i.e. printing progress)")
    parser.add_argument("--steps", type=int, default=10000000,
                        help="Number of training steps")
    parser.add_argument("--regenerate_steps", type=int, default=1000,
                        help="How often to regenerate training set (to avoid "
                        "overfitting). Ignored if <= 0.")
    parser.add_argument("--buffer_size", type=int, default=100,
                        help="Buffer size for data iterator shuffler. Affects"
                        " how much system memory is required.")
    # Optimization
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--rate_loss_weight", type=float, default=1.,
                        help="Weight on KL loss term (analogous to information"
                        " 'rate').")
    parser.add_argument("--beta0", type=float, default=None,
                        help="For gradually increasing the rate loss weight "
                        "(i.e. 'beta'). This is the starting value of beta")
    parser.add_argument("--beta_steps", type=int, default=200000,
                        help="When gradually increasing the rate loss weight "
                        "(i.e. 'beta'), beta is increased linearly with number"
                        "of training steps. This is the step count at which "
                        "beta becomes equal to 'rate_loss_weight'")
    parser.add_argument("--reconstruction_loss_weights", type=float, nargs="*",
                        default=[1., 1.],
                        help="How much to weight reconstruction loss term in "
                        "total loss, separated by encoder (first value) and "
                        "decoder (second value)")
    parser.add_argument("--decision_loss_weights", type=float, nargs="*",
                        default=[1., 1.],
                        help="How much to weight decision loss term in total "
                        "loss.")
    parser.add_argument("--regularizer_loss_weight", type=float, default=1e-8,
                        help="How much to weight regularizer loss in total "
                        "loss.")
    parser.add_argument("--dropout_prob", type=float, default=0.5,
                        help="Probability parameter in tf.nn.dropout")
    # Architecture
    parser.add_argument("--activation", type=str, default='relu',
                        help="Activation for hidden units")
    parser.add_argument("--sampling_off", action="store_true",
                        help="Turn off sampling step in channel.")
    # Task
    parser.add_argument("--dataset", type=str, default='fruits360',
                        help="Which dataset to train on.")
    return parser


def make_net(args):
    net = VAE(args.activation, args.checkpoint_dir, args.trainingset_dir,
              dataset=args.dataset,
              batch_size=args.batch,
              buffer_size=args.buffer_size,
              regenerate_steps=args.regenerate_steps,
              w_reconstruction=args.reconstruction_loss_weights,
              w_rate=args.rate_loss_weight,
              beta0=args.beta0,
              beta_steps=args.beta_steps,
              w_decision=args.decision_loss_weights,
              w_reg=args.regularizer_loss_weight,
              dropout_prob=args.dropout_prob,
              sampling_off=args.sampling_off)
    return net


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_parser_args(parser)
    args = parser.parse_args()
    vae = make_net(args)
    vae.build()
    vae.train(args.steps, args.report, args.learning_rate)
