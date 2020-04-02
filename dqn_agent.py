import os
import random
import numpy as np
import tensorflow as tf
import replay_buffer


def conv_net(scope, inputs):
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
        return layer_out


def dense_net(scope, inputs, hidden_size, output_size):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        pre_layer = tf.layers.dense(inputs, hidden_size, use_bias=True,
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    activation=tf.nn.relu, name='PRE_LAYER_1')

        adv_layer_1 = tf.layers.dense(pre_layer, output_size, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=None, name='ADV_LAYER_1')

        val_layer_1 = tf.layers.dense(pre_layer, 1, use_bias=True,
                                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                      activation=None, name='VAL_LAYER_1')

        advantage = (adv_layer_1 - tf.reduce_mean(adv_layer_1, axis=-1, keepdims=True))
        value = tf.tile(val_layer_1, [1, output_size])
        output = advantage + value
        return output


class DQNAgent(object):
    def __init__(self, id, config, session, type):
        self.id = id
        self.name = 'AGENT_' + type.upper() + '_' + str(id)
        self.session = session
        self.type = type

        # Extract relevant configuration:
        self.config = {}
        self.config['env_n_actions'] = config['env_n_actions']
        self.config['env_obs_dims'] = config['env_obs_dims']
        self.config['env_type'] = config['env_type']

        dqn_config_params = [
            'dqn_gamma',
            'dqn_rm_init',
            'dqn_rm_max',
            'dqn_target_update',
            'dqn_batch_size',
            'dqn_learning_rate',
            'dqn_train_period',
            'dqn_adam_epsilon',
            'dqn_epsilon_start',
            'dqn_epsilon_final',
            'dqn_epsilon_steps',
            'dqn_huber_loss_delta'
        ]

        for param in dqn_config_params:
            self.config[param] = config[param]

        self.epsilon = self.config['dqn_epsilon_start']
        self.epsilon_step_size = (self.config['dqn_epsilon_start'] - self.config['dqn_epsilon_final']) \
                                 / self.config['dqn_epsilon_steps']

        # Scoped names
        self.name_online = self.name + '/' + 'DQN_ONLINE'
        self.name_target = self.name + '/' + 'DQN_TARGET'

        self.obs, self.q_values, self.evaluation, self.latent_features = \
            self.build_model(self.name_online, self.config['env_n_actions'])

        self.obs_target, self.q_values_target, self.evaluation_target, self.latent_features_target = \
            self.build_model(self.name_target, self.config['env_n_actions'])

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_online)
        trainable_vars_by_name = {var.name[len(self.name_online):]: var for var in trainable_vars}

        trainable_vars_t = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name_target)
        trainable_vars_by_name_t = {var.name[len(self.name_target):]: var for var in trainable_vars_t}

        copy_ops = [target_var.assign(trainable_vars_by_name[var_name])
                    for var_name, target_var in trainable_vars_by_name_t.items()]
        self.update_target_weights = tf.group(*copy_ops)

        self.action, self.td_target, self.td_error, self.loss, self.grads_update = self.build_training_ops()

        self.replay_memory = replay_buffer.ReplayBuffer(self.config['dqn_rm_max'])

        # --------------------------------------------------------------------------------------------------------------

        self.post_init_steps = 0
        self.training_steps = 0
        self.n_episode = 0

        self.sample_loss_mean = 0.0
        self.sample_n = 0.0

        self.self_loss_mean = 0.0
        self.self_loss_n = 0.0
        self.expert_loss_mean = 0.0
        self.expert_loss_n = 0.0
        self.ep_r_steps = 0

        self.rnd_rm = None

    # ------------------------------------------------------------------------------------------------------------------

    def feedback_observe(self, obs, action, reward, obs_next, done, source=None, state=None):
        if done:
            self.n_episode += 1
        old_transition = self.replay_memory.add(obs, action, reward, obs_next, done, source, state)
        return old_transition

    # ------------------------------------------------------------------------------------------------------------------

    def feedback_learn(self):
        loss = 0.0
        td_error_batch = [0.0]

        perform_learning = self.replay_memory.__len__() >= self.config['dqn_rm_init']

        if perform_learning:
            self.ep_r_steps += 1
            if self.epsilon > self.config['dqn_epsilon_final']:
                self.epsilon -= self.epsilon_step_size

            self.post_init_steps += 1
            if self.post_init_steps % self.config['dqn_train_period'] == 0:

                td_error_batch, loss = self.train_model()

                if self.training_steps >= self.config['dqn_target_update']:
                    self.training_steps = 0
                    self.session.run(self.update_target_weights)

        return td_error_batch, loss

    # ------------------------------------------------------------------------------------------------------------------

    def act(self, obs, evaluation=False):
        if evaluation:
            return self.greedy_action(obs, True)
        else:
            return self.epsilon_greedy_action(obs, False)

    def epsilon_greedy_action(self, obs, evaluation=False):
        if random.random() < self.epsilon:
            return self.random_action()
        else:
            return self.greedy_action(obs, evaluation)

    def greedy_action(self, obs, evaluation=False):
        qv = self.get_q_values(obs, evaluation)
        return np.argmax(qv)

    def random_action(self):
        return random.randrange(self.config['env_n_actions'])

    # ------------------------------------------------------------------------------------------------------------------

    def build_model(self, name, output_size):
        obs = None
        if self.config['env_type'] == -1:
            obs = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0]], name=name + '_OBS')
        elif self.config['env_type'] == 0 or self.config['env_type'] == 1:
            obs = tf.placeholder(tf.float32, [None, self.config['env_obs_dims'][0],
                                              self.config['env_obs_dims'][1],
                                              self.config['env_obs_dims'][2]], name=name + '_OBS')

        evaluation = tf.placeholder(tf.bool, name=name + '_EVALUATION')

        latent_features = None
        if self.config['env_type'] == -1:
            latent_features = obs
        elif self.config['env_type'] == 0 or self.config['env_type'] == 1:
            latent_features = conv_net(name, obs)

        q_values = dense_net(name, latent_features, 128, output_size)

        return obs, q_values, evaluation, latent_features

    # ------------------------------------------------------------------------------------------------------------------

    def build_training_ops(self):
        # Placeholders
        action = tf.placeholder(tf.int32, [None], name='ACTIONS_' + str(self.id))
        td_target = tf.placeholder(tf.float32, [None], name='LABELS_' + str(self.id))

        # Operations
        action_one_hot = tf.one_hot(action, self.config['env_n_actions'], 1.0, 0.0)
        q_values_reduced = tf.reduce_sum(tf.multiply(self.q_values, action_one_hot), reduction_indices=1)

        td_error = tf.abs(td_target - q_values_reduced)
        loss = tf.losses.huber_loss(labels=td_target, predictions=q_values_reduced,
                                    delta=self.config['dqn_huber_loss_delta'])

        optimizer = tf.train.AdamOptimizer(self.config['dqn_learning_rate'], epsilon=self.config['dqn_adam_epsilon'])
        grads_update = optimizer.minimize(loss)

        return action, td_target, td_error, loss, grads_update

    # ------------------------------------------------------------------------------------------------------------------

    def get_q_values(self, obs, evaluation=False):
        feed_dict = {self.obs: [obs.astype(dtype=np.float32)],
                     self.evaluation: evaluation}
        return self.session.run(self.q_values, feed_dict=feed_dict)

    # ------------------------------------------------------------------------------------------------------------------

    def get_latent_features(self, obs, evaluation=False):
        feed_dict = {self.obs: [obs.astype(dtype=np.float32)],
                     self.evaluation: evaluation}
        return self.session.run(self.latent_features, feed_dict=feed_dict)

    # ------------------------------------------------------------------------------------------------------------------

    def fix_batch_form(self, var, is_batch):
        return var if is_batch else [var]

    # ------------------------------------------------------------------------------------------------------------------

    def get_td_target(self, reward_batch_in, obs_next_batch_in, done_batch_in, evaluation=False):

        # Transform single sample into batch structure
        is_batch = isinstance(reward_batch_in, list) or isinstance(reward_batch_in, np.ndarray)

        obs_next_batch = obs_next_batch_in if isinstance(obs_next_batch_in, list) \
            else obs_next_batch_in.astype(dtype=np.float32)

        reward_batch = self.fix_batch_form(reward_batch_in, is_batch)
        obs_next_batch = self.fix_batch_form(obs_next_batch, is_batch)
        done_batch = self.fix_batch_form(done_batch_in, is_batch)

        # Compute td-target(s)
        feed_dict = {self.obs: obs_next_batch,
                     self.obs_target: obs_next_batch,
                     self.evaluation: evaluation,
                     self.evaluation_target: evaluation}

        q_values_next_batch, q_values_next_target_batch = \
            self.session.run([self.q_values, self.q_values_target], feed_dict=feed_dict)

        action_next_batch = np.argmax(q_values_next_batch, axis=1)

        td_target_batch = []  # TD Target
        for j in range(len(reward_batch)):
            td_target = reward_batch[j] + (1.0 - done_batch[j]) * self.config['dqn_gamma'] * \
                        q_values_next_target_batch[j][action_next_batch[j]]
            td_target_batch.append(td_target)

        return td_target_batch if is_batch else td_target_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def arrange_feed_dict(self, obs_batch_in, action_batch_in, reward_batch_in, obs_next_batch_in, done_batch_in,
                          evaluation=False):

        # Transform single sample into batch structure
        is_batch = isinstance(reward_batch_in, list) or isinstance(reward_batch_in, np.ndarray)

        obs_batch = obs_batch_in if isinstance(obs_batch_in, list) \
            else obs_batch_in.astype(dtype=np.float32)

        obs_next_batch = obs_next_batch_in if isinstance(obs_next_batch_in, list) \
            else obs_next_batch_in.astype(dtype=np.float32)

        obs_batch = self.fix_batch_form(obs_batch, is_batch)
        action_batch = self.fix_batch_form(action_batch_in, is_batch)
        reward_batch = self.fix_batch_form(reward_batch_in, is_batch)
        obs_next_batch = self.fix_batch_form(obs_next_batch, is_batch)
        done_batch = self.fix_batch_form(done_batch_in, is_batch)

        td_target_batch = self.get_td_target(reward_batch, obs_next_batch, done_batch, evaluation)

        feed_dict = {self.obs: obs_batch,
                     self.action: action_batch,
                     self.td_target: td_target_batch,
                     self.evaluation: False}

        return feed_dict, is_batch

    # ------------------------------------------------------------------------------------------------------------------

    def get_td_error(self, obs_batch, action_batch, reward_batch, obs_next_batch, done_batch, evaluation=False):
        feed_dict, is_batch = self.arrange_feed_dict(obs_batch, action_batch, reward_batch, obs_next_batch, done_batch,
                                                     evaluation)
        td_error_batch = self.session.run(self.td_error, feed_dict=feed_dict)
        return td_error_batch if is_batch else td_error_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def get_loss(self, obs_batch, action_batch, reward_batch, obs_next_batch, done_batch, evaluation=False):
        feed_dict, is_batch = self.arrange_feed_dict(obs_batch, action_batch, reward_batch, obs_next_batch, done_batch,
                                                     evaluation)
        loss_batch = self.session.run(self.loss, feed_dict=feed_dict)
        return loss_batch if is_batch else loss_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def get_grads_update(self, obs_batch, action_batch, reward_batch, obs_next_batch, done_batch, evaluation=False):
        feed_dict, is_batch = self.arrange_feed_dict(obs_batch, action_batch, reward_batch, obs_next_batch, done_batch,
                                                     evaluation)
        td_error_batch, loss_batch, _ = self.session.run([self.td_error, self.loss, self.grads_update], feed_dict=feed_dict)
        return td_error_batch if is_batch else td_error_batch[0], loss_batch if is_batch else loss_batch[0]

    # ------------------------------------------------------------------------------------------------------------------

    def train_model(self):
        self.training_steps += 1

        minibatch = self.replay_memory.sample(self.config['dqn_batch_size'])

        obs_batch = minibatch[0]
        action_batch = minibatch[1]
        reward_batch = minibatch[2]
        obs_next_batch = minibatch[3]
        done_batch = minibatch[4]
        source_batch = minibatch[5]

        td_error, loss = self.get_grads_update(obs_batch, action_batch, reward_batch, obs_next_batch, done_batch,
                                               evaluation=False)

        return td_error, loss

    def save_model(self, saver, models_dir, session_name, checkpoint):
        model_path = os.path.join(os.path.join(models_dir, session_name), 'model-{}.ckpt').format(checkpoint)
        print('[{}] Saving model... {}'.format(checkpoint, model_path))
        saver.save(self.session, model_path)

    def restore(self, models_dir, session_name, checkpoint):
        print('Restoring...')
        print('Scope: {}'.format(self.name))
        print('# of variables: {}'.format(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))))
        loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))
        loader.restore(self.session,
                       os.path.join(os.path.join(models_dir, session_name), 'model-' + str(checkpoint) + '.ckpt'))
