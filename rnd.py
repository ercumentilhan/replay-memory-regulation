import numpy as np
import tensorflow as tf

class RND(object):
    def __init__(self, id, config, session, lr, adam_eps):
        self.id = id
        self.name = 'AGENT_' + str(id)
        self.session = session

        # Extract relevant configuration:
        self.config = {}
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_type'] = config['env_type']
        self.config['env_name'] = config['env_name']

        self.config['rnd_learning_rate'] = lr
        self.config['rnd_adam_epsilon'] = adam_eps

        # Hyperparameters
        self.config['rnd_output_size'] = 6
        self.config['rnd_hidden_size'] = 128
        self.config['rnd_obs_norm_max_n'] = 5000 if self.config['env_type'] == 1 else 1000

        normalization_coefficients = self.get_normalization_coefficients()
        self.obs_mean = np.array(normalization_coefficients[0])
        self.obs_std = np.array(normalization_coefficients[1])

        self.name_fixed = self.name + '/RND_FIXED'
        self.name_online = self.name + '/RND_ONLINE'

        self.input_fixed, self.output_fixed, self.evaluation_fixed = \
            self.build_model(self.name_fixed, self.config['rnd_hidden_size'], self.config['rnd_output_size'])

        self.input_online, self.output_online, self.evaluation_online = \
            self.build_model(self.name_online, self.config['rnd_hidden_size'], self.config['rnd_output_size'])

        self.labels, \
        self.losses, \
        self.minimises, \
        self.error = self.build_training_op()

        self.training_steps = 0

    # ------------------------------------------------------------------------------------------------------------------

    def dense_net(self, scope, inputs, hidden_size, output_size):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.dense(inputs, hidden_size, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=tf.nn.relu, name='DENSE_LAYER_1')

            layer_2 = tf.layers.dense(layer_1, output_size, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=None, name='DENSE_LAYER_2')

            return layer_2

    def convolutional_net(self, scope, inputs):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            layer_1 = tf.layers.conv2d(inputs=inputs,
                                       filters=16,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='VALID',
                                       kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                       activation=tf.nn.relu,
                                       name='CONV_LAYER_1')

            layer_out = tf.contrib.layers.flatten(layer_1)
            print(layer_out.get_shape())
            return layer_out

    # ------------------------------------------------------------------------------------------------------------------

    def build_model(self, name, dense_hidden_size, output_size):
        input = None
        if self.config['env_type'] == -1:
            input = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0]], name=name + '_OBS')
        elif self.config['env_type'] == 0 or self.config['env_type'] == 1:
            input = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0],
                                              self.config['env_obs_dims'][1],
                                              self.config['env_obs_dims'][2]], name=name + '_OBS')

        evaluation = tf.placeholder(tf.bool, name=name + '_EVALUATION')

        latent_features = None
        if self.config['env_type'] == -1:
            latent_features = input
        elif self.config['env_type'] == 0 or self.config['env_type'] == 1:
            latent_features = self.convolutional_net(name, input)

        output = self.dense_net(name, latent_features, dense_hidden_size, output_size)

        return input, output, evaluation

    # ------------------------------------------------------------------------------------------------------------------

    def build_training_op(self):
        label = tf.placeholder(tf.float32, [None, self.config['rnd_output_size']], name='LABELS_' + str(self.id))
        error = tf.abs(label - self.output_online)

        optimizer = tf.train.AdamOptimizer(self.config['rnd_learning_rate'], epsilon=self.config['rnd_adam_epsilon'])

        losses = []
        losses.append(tf.losses.mean_squared_error(labels=label, predictions=self.output_online))
        losses.append(tf.square(1.0 - tf.losses.absolute_difference(labels=label, predictions=self.output_online)))

        minimises = []
        for loss in losses:
            minimises.append(optimizer.minimize(loss))

        return label, losses, minimises, error

    # ------------------------------------------------------------------------------------------------------------------

    def train_model(self, obs_batch_in, loss_id, is_batch=True, normalize=True):
        self.training_steps += 1

        if normalize:
            obs_batch_in = self.normalize_obs(obs_batch_in)

        obs_batch = obs_batch_in if isinstance(obs_batch_in, list) else obs_batch_in
        obs_batch = obs_batch if is_batch else [obs_batch]

        feed_dict = {self.input_fixed: obs_batch, self.evaluation_fixed: False}

        output_fixed = self.session.run([self.output_fixed], feed_dict=feed_dict)[0]

        feed_dict = {self.input_online: obs_batch, self.evaluation_online: False,
                     self.labels: output_fixed}

        loss, _, error = self.session.run([self.losses[loss_id], self.minimises[loss_id], self.error],
                                          feed_dict=feed_dict)

        return loss, np.abs(error)

    # ------------------------------------------------------------------------------------------------------------------

    def get_error(self, obs_in, evaluation=False, normalize=True):

        obs = self.normalize_obs(obs_in) if normalize else obs_in

        feed_dict = {self.input_fixed: [obs],
                     self.input_online: [obs],
                     self.evaluation_fixed: evaluation,
                     self.evaluation_online: evaluation}

        output_fixed, output_online = self.session.run([self.output_fixed, self.output_online], feed_dict=feed_dict)

        return ((output_fixed - output_online) ** 2).mean(axis=1)[0]

    # ------------------------------------------------------------------------------------------------------------------

    def normalize_obs(self, obs):
        return np.clip(((obs - self.obs_mean) / self.obs_std), -5, 5)

    # ------------------------------------------------------------------------------------------------------------------

    # Pre-computed values for RND observation normalisation
    def get_normalization_coefficients(self):
        if self.config['env_type'] == 0:
            if self.config['rnd_obs_norm_max_n'] == 1000:
                return [0.00826446, 0.00826446, 0.44628099], [0.09053265, 0.09053265, 0.49710589]

        elif self.config['env_type'] == 1:
            if self.config['env_name'] == 'asterix':
                if self.config['rnd_obs_norm_max_n'] == 5000:
                    return [0.01, 0.017792, 0.025028, 0.007236], \
                           [0.09949878, 0.11056552, 0.1374219, 0.05821605]

            elif self.config['env_name'] == 'freeway':
                if self.config['rnd_obs_norm_max_n'] == 5000:
                    return [0.01, 0.08, 0.02599799, 0.02401199, 0.003992, 0.01800398, 0.007994], \
                           [0.09949875, 0.27129334, 0.10334328, 0.10752366, 0.11138161, 0.12050621,
                            0.10539624]

            elif self.config['env_name'] == 'seaquest':
                if self.config['rnd_obs_norm_max_n'] == 5000:
                    return [0.01, 0.01, 0.00372, 0.00883598, 0.00069, 0.00462199, 0.002006, 0.08390398, 0.000628, 0.00420401], \
                           [0.09949882, 0.09949882, 0.03678932, 0.05520402, 0.00679462, 0.03766833, 0.01905091, 0.27467063, 0.00624852, 0.03574077]

