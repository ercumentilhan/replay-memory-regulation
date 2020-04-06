import os
from time import localtime, strftime
import pathlib
import glob
import shutil
import random
import numpy as np
import tensorflow as tf
import cv2
from dqn_agent import DQNAgent
from statistics import Statistics
from rnd import RND

os.environ['TF_CPP_MIN_LONG_LEVEL'] = '2'


class Executor:
    def __init__(self, config, env, eval_env):
        self.config = config
        self.env = env
        self.eval_env = eval_env

        self.stats = None

        self.student_agent = None

        self.evaluation_dir = None
        self.save_videos_path = None

        self.steps_reward = 0.0
        self.steps_error_in = 0.0
        self.steps_error_out = 0.0

        self.episode_duration = 0
        self.episode_reward = 0.0
        self.episode_error_in = 0.0
        self.episode_error_out = 0.0

        self.episode_visited = set()

        self.obs_images = None
        self.tr_info = None

        # ==============================================================================================================

        self.process = None
        self.run_id = None

        self.scripts_dir = None
        self.local_workspace_dir = None

        self.runs_local_dir = None
        self.summaries_dir = None
        self.checkpoints_dir = None
        self.copy_scripts_dir = None
        self.videos_dir = None

        self.save_summary_path = None
        self.save_model_path = None
        self.save_scripts_path = None
        self.save_videos_path = None

        self.plots_subdirs = None
        self.save_plots_paths = None

        self.session = None
        self.summary_writer = None
        self.saver = None

        self.teacher_agent = None

        self.rnd_rm = None

        self.rnd_uncertainty_c = None
        self.rnd_uncertainty_d = None

        self.action_advising_enabled = None
        self.action_advising_budget = None
        self.action_advising_method = None
        self.action_advising_rm_th = self.config['action_advising_rm_th']
        self.action_advising_check_rm = None

        # RND Observation normalization
        self.obs_running_mean = None
        self.obs_running_std = None
        self.obs_norm_n = 0
        self.obs_norm_max_n = 5000 if self.config['env_type'] == 1 else 1000

        self.data_collection_period = 500 if self.config['env_type'] == 1 else 100  # Frames
        self.data_collection_step = 0
        self.data_rnd_uncertainty = None
        self.data_rnd_rm = None

        # Online counters
        self.rm_obs_counter_all = None
        self.rm_obs_counter_teacher = None
        self.rm_tr_counter_all = None
        self.rm_tr_counter_teacher = None

        # Snapshots
        self.data_rm_obs_counter_all = None
        self.data_rm_obs_counter_teacher = None
        self.data_rm_tr_counter_all = None
        self.data_rm_tr_counter_teacher = None

    # ------------------------------------------------------------------------------------------------------------------

    def render(self, env):
        if self.config['env_type'] == 0:
            return env.render()
        elif self.config['env_type'] == 1:
            return env.render_state()

    # ------------------------------------------------------------------------------------------------------------------

    def run(self):
        os.environ['PYTHONHASHSEED'] = str(self.config['seed'])
        random.seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        tf.set_random_seed(self.config['seed'])

        self.run_id = self.config['run_id'] if self.config['run_id'] is not None \
            else strftime("%Y%m%d-%H%M%S", localtime()) + '-' + str(self.config['process_index'])
        self.seed_id = str(self.config['seed'])

        print('Run ID: {}'.format(self.run_id))

        self.scripts_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_workspace_dir = os.path.join(str(pathlib.Path(self.scripts_dir).parent.parent.parent.parent))

        print('{} (Scripts directory)'.format(self.scripts_dir))
        print('{} (Local Workspace directory)'.format(self.local_workspace_dir))

        self.runs_local_dir = os.path.join(self.local_workspace_dir, 'Runs')
        os.makedirs(self.runs_local_dir, exist_ok=True)

        self.summaries_dir = os.path.join(self.runs_local_dir, 'Summaries')
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.checkpoints_dir = os.path.join(self.runs_local_dir, 'Checkpoints')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.copy_scripts_dir = os.path.join(self.runs_local_dir, 'Scripts')
        os.makedirs(self.copy_scripts_dir, exist_ok=True)

        self.videos_dir = os.path.join(self.runs_local_dir, 'Videos')
        os.makedirs(self.videos_dir, exist_ok=True)

        self.plots_dir = os.path.join(self.runs_local_dir, 'Plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        self.data_dir = os.path.join(self.runs_local_dir, 'Data')
        os.makedirs(self.data_dir, exist_ok=True)

        # --------------------------------------------------------------------------------------------------------------

        self.save_summary_path = os.path.join(self.summaries_dir, self.run_id, self.seed_id)
        self.save_model_path = os.path.join(self.checkpoints_dir, self.run_id, self.seed_id)
        self.save_scripts_path = os.path.join(self.copy_scripts_dir, self.run_id, self.seed_id)
        self.save_videos_path = os.path.join(self.videos_dir, self.run_id, self.seed_id)
        self.save_data_path = os.path.join(self.data_dir, self.run_id, self.seed_id)

        self.plots_subdirs = []
        self.plots_subdirs.append(os.path.join(self.plots_dir, 'TD-Error-All'))  # 0
        self.plots_subdirs.append(os.path.join(self.plots_dir, 'State-Uncertainty'))  # 1
        self.plots_subdirs.append(os.path.join(self.plots_dir, 'Combined'))  # 2
        self.plots_subdirs.append(os.path.join(self.plots_dir, 'ER'))  # 3
        for graphs_subdir in self.plots_subdirs:
            os.makedirs(graphs_subdir, exist_ok=True)

        self.save_plots_paths = [os.path.join(plots_subdir, self.run_id, self.seed_id) for plots_subdir in
                                 self.plots_subdirs]

        for save_plots_path in self.save_plots_paths:
            os.makedirs(save_plots_path, exist_ok=True)

        if self.config['save_models']:
            os.makedirs(self.save_model_path, exist_ok=True)
        os.makedirs(self.save_videos_path, exist_ok=True)
        os.makedirs(self.save_data_path, exist_ok=True)

        self.copy_scripts(self.save_scripts_path)

        if self.config['use_gpu']:
            print('Using GPU.')
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1)
        else:
            print('Not using GPU.')
            session_config = tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True,
                device_count={'CPU': 1, 'GPU': 0})

        self.session = tf.InteractiveSession(graph=tf.get_default_graph(), config=session_config)
        self.summary_writer = tf.summary.FileWriter(self.save_summary_path, self.session.graph)

        # --------------------------------------------------------------------------------------------------------------

        # Experiment setup format: abc (0: No Advising)
        # a: Action advising method
        # -- 1: Early advising
        # -- 2: Uniformly random advising
        # -- 3: Uncertainty based advising (RND)
        # -- 4: Uncertainty based bootstrapped DQN
        # b: Replay memory checking (True/False)
        # c: Budget

        if self.config['experiment_setup'] == 0:  # Self exploration
            self.action_advising_enabled = False
            self.action_advising_check_rm = False
            self.action_advising_budget = 0

        else:
            es_1 = self.config['experiment_setup']
            es_2 = es_1 % 100
            es_3 = es_2 % 10

            action_advising_budgets = {
                0: 500,
                1: 1000,
                2: 2500,
                3: 5000,
                4: 10000,
                5: 25000,
                6: 50000,
                7: 100000
            }

            self.action_advising_enabled = True
            self.action_advising_method = es_1 // 100
            self.action_advising_check_rm = es_2 // 10
            self.action_advising_budget = action_advising_budgets[es_3]

        # --------------------------------------------------------------------------------------------------------------

        # Config to be passed to agents
        if self.config['env_type'] == 0:
            self.config['env_obs_dims'] = self.env.obs_space.shape
            self.config['env_n_actions'] = self.env.action_space.n
        elif self.config['env_type'] == 1:
            self.config['env_obs_dims'] = self.env.state_shape()
            self.config['env_n_actions'] = self.env.num_actions()

        student_agent_name = self.run_id.replace('-', '') + '0' + '_' + str(self.config['seed'])

        self.student_agent = DQNAgent(student_agent_name, self.config, self.session, 'task')
        self.config['student_model_name'] = self.student_agent.name
        print('Student agent name: {}'.format(self.student_agent.name))

        self.save_config(self.config, os.path.join(self.save_summary_path, 'config.txt'))

        if self.config['env_type'] == 1 and self.config['experiment_setup'] != 0:
            self.teacher_agent = DQNAgent(self.config['expert_agent_id'].replace("-", "") + '0_'
                                          + self.config['expert_agent_seed'], self.config, self.session, 'task')
            print('Expert agent name: {}'.format(self.teacher_agent.name))

        # --------------------------------------------------------------------------------------------------------------

        # Initialize RND:
        if self.action_advising_check_rm == 2:  # RM checking with RND
            self.rnd_rm = RND(student_agent_name + '_RM', self.config, self.session,
                              self.config['rm_rnd_learning_rate'],
                              self.config['rm_rnd_adam_epsilon'])
            self.student_agent.rnd_rm = self.rnd_rm

        # --------------------------------------------------------------------------------------------------------------

        if self.config['env_type'] == 0:
            n_data_points = int(self.config['n_training_frames'] / self.data_collection_period) + 1

            self.data_rnd_uncertainty = \
                np.zeros((self.config['env_obs_dims'][0], self.config['env_obs_dims'][1], n_data_points), dtype=np.float32)
            self.data_rnd_rm = \
                np.zeros((self.config['env_obs_dims'][0], self.config['env_obs_dims'][1], n_data_points), dtype=np.float32)

            # Online counters
            self.rm_obs_counter_all = np.zeros(self.env.n_states, dtype=int)
            self.rm_obs_counter_teacher = np.zeros(self.env.n_states, dtype=int)
            self.rm_tr_counter_all = np.zeros(self.env.n_transitions, dtype=int)
            self.rm_tr_counter_teacher = np.zeros(self.env.n_transitions, dtype=int)

            # Snapshots
            self.data_rm_obs_counter_all = np.zeros((self.env.n_states, n_data_points), dtype=int)
            self.data_rm_obs_counter_teacher = np.zeros((self.env.n_states, n_data_points), dtype=int)
            self.data_rm_tr_counter_all = np.zeros((self.env.n_transitions, n_data_points), dtype=int)
            self.data_rm_tr_counter_teacher = np.zeros((self.env.n_transitions, n_data_points), dtype=int)

        # --------------------------------------------------------------------------------------------------------------

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Number of parameters: {}'.format(total_parameters))

        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

        self.stats = Statistics(self.summary_writer, self.session)

        # Restore
        if self.config['experiment_setup'] != 0:
            if self.config['env_type'] == 1:
                self.teacher_agent.restore(self.checkpoints_dir,
                                           self.config['expert_agent_id'] + '/' + self.config['expert_agent_seed'],
                                           self.config['expert_agent_checkpoint'])

        if not self.config['save_models']:
            tf.get_default_graph().finalize()

        reward_is_seen = False

        if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
            eval_score = self.evaluate()
            print('Evaluation @ {} | {}'.format(self.stats.n_env_steps, eval_score))

        render = self.stats.n_episodes % self.config['visualization_period'] == 0

        if render:
            self.obs_images = []
            self.tr_info = []

        obs = None
        if self.config['env_type'] == 0:
            obs = self.env.reset()
        elif self.config['env_type'] == 1:
            self.env.reset()
            obs = self.env.state().astype(dtype=np.float32)

        if render:
            self.obs_images.append(self.render(self.env))

        if self.config['env_type'] == 0:
            self.record_data()

        while True:
            if self.obs_norm_n < self.obs_norm_max_n:
                obs_mean = obs.mean(axis=(0, 1))
                obs_std = obs.std(axis=(0, 1))
                if self.obs_norm_n == 0:
                    self.obs_running_mean = obs_mean
                    self.obs_running_std = obs_std
                else:
                    self.obs_running_mean = \
                        self.obs_running_mean + (obs_mean - self.obs_running_mean)/(self.obs_norm_n + 1)
                    self.obs_running_std = \
                        self.obs_running_std + (obs_std - self.obs_running_std) / (self.obs_norm_n + 1)
                self.obs_norm_n += 1

            if self.obs_norm_n == self.obs_norm_max_n:
                print(self.obs_running_mean)
                print(self.obs_running_std)
                self.obs_norm_n += 1
            #

            state_id = self.env.state_id_dict[(self.env.state.agent_pos[0], self.env.state.agent_pos[1])] \
                if self.config['env_type'] == 0 else None

            # ----------------------------------------------------------------------------------------------------------
            # Action Advising

            get_action_advice = False
            if self.action_advising_enabled and self.action_advising_budget > 0:
                if self.action_advising_method == 1:
                    get_action_advice = True
                elif self.action_advising_method == 2:
                    if random.random() < 0.5:
                        get_action_advice = True

            # Second-factor check for RM
            if get_action_advice and self.action_advising_check_rm != 0:
                if self.action_advising_check_rm == 1:
                    if self.rm_obs_counter_teacher[state_id] >= self.action_advising_rm_th:
                        get_action_advice = False
                elif self.action_advising_check_rm == 2:
                    sparsity = self.rnd_rm.get_error(obs, normalize=True)
                    if sparsity >= self.action_advising_rm_th:
                        pass
                    else:
                        get_action_advice = False

            if get_action_advice:
                self.action_advising_budget -= 1
                self.stats.advices_taken += 1
                self.stats.advices_taken_cumulative += 1

                if self.config['env_type'] == 0:
                    action = self.env.optimal_action()
                elif self.config['env_type'] == 1:
                    action = self.teacher_agent.greedy_action(obs, evaluation=True)

                source = 1
            else:
                action = self.student_agent.act(obs, evaluation=False)
                source = 0

            # ----------------------------------------------------------------------------------------------------------

            transition_id = self.env.transition_id_dict[(self.env.state.agent_pos[0], self.env.state.agent_pos[1], action)] \
                if self.config['env_type'] == 0 else None

            obs_next, reward, done = None, None, None
            if self.config['env_type'] == 0:
                obs_next, reward, done = self.env.step(action)
            elif self.config['env_type'] == 1:
                reward, done = self.env.act(action)
                obs_next = self.env.state().astype(dtype=np.float32)

            td_error = self.student_agent.get_td_error(obs, action, reward, obs_next, done)

            if render:
                self.obs_images.append(self.render(self.env))

            self.episode_error_in += td_error
            self.episode_reward += reward
            self.episode_duration += 1

            self.steps_error_in += td_error
            self.steps_reward += reward
            self.stats.n_env_steps += 1

            if reward > 0 and reward_is_seen is False:
                reward_is_seen = True
                print(">>> Reward is seen at ", self.stats.n_episodes, "|", self.episode_duration)

            if source == 1:
                if self.action_advising_check_rm == 2:
                    self.rnd_rm.train_model(obs, loss_id=0, is_batch=False, normalize=True)

            if self.config['env_type'] == 0:
                self.rm_obs_counter_all[state_id] += 1
                self.rm_tr_counter_all[transition_id] += 1
                if source == 1:
                    self.rm_obs_counter_teacher[state_id] += 1
                    self.rm_tr_counter_teacher[transition_id] += 1

            # ----------------------------------------------------------------------------------------------------------
            # Dropped data from RM

            old_transition = self.student_agent.feedback_observe(obs, action, reward, obs_next, done, source, state_id)

            if old_transition is not None:
                if self.action_advising_check_rm == 2 and old_transition[5] == 1:
                    self.rnd_rm.train_model(old_transition[0], loss_id=1, is_batch=False, normalize=True)

                if self.config['env_type'] == 0:
                    old_state_id = old_transition[6]
                    old_action = old_transition[1]
                    old_agent_pos = self.env.agent_pos_dict[old_state_id]
                    old_transition_id = self.env.transition_id_dict[(old_agent_pos[0], old_agent_pos[1], old_action)]

                    self.rm_obs_counter_all[old_state_id] -= 1
                    self.rm_tr_counter_all[old_transition_id] -= 1

                    if old_transition[5] == 1:
                        self.rm_obs_counter_teacher[old_state_id] -= 1
                        self.rm_tr_counter_teacher[old_transition_id] -= 1

            # ----------------------------------------------------------------------------------------------------------

            td_error_batch, loss = self.student_agent.feedback_learn()
            td_error_batch_sum = np.sum(td_error_batch)

            self.episode_error_out += td_error_batch_sum
            self.steps_error_out += td_error_batch_sum

            self.stats.loss += loss
            obs = obs_next

            if self.config['env_type'] == 0 and self.stats.n_env_steps % self.data_collection_period == 0:
                self.record_data()

            if done:
                self.action_advising_countdown = 0

                self.stats.n_episodes += 1
                self.stats.episode_reward_auc += np.trapz([self.stats.episode_reward_last, self.episode_reward])
                self.stats.episode_reward_last = self.episode_reward

                self.stats.update_summary_episode(self.episode_reward, self.stats.episode_reward_auc,
                                                  self.episode_duration,self.episode_error_in, self.episode_error_out)

                print('ER: {:.1f} ({}) (error: {:.3f}) @ {} frames - {}'
                     .format(self.episode_reward, self.stats.n_episodes, self.episode_error_in, self.stats.n_env_steps,
                             self.stats.advices_taken_cumulative))

                if render:
                    self.write_video(self.obs_images, '{}_{}'.format(str(self.stats.n_episodes - 1),
                                                                           str(self.stats.n_env_steps - self.episode_duration)))
                    self.obs_images.clear()
                    self.tr_info.clear()

                self.episode_duration = 0
                self.episode_reward = 0.0
                self.episode_error_in = 0.0
                self.episode_error_out = 0.0

                render = self.stats.n_episodes % self.config['visualization_period'] == 0

                obs = None
                if self.config['env_type'] == 0:
                    obs = self.env.reset()
                elif self.config['env_type'] == 1:
                    self.env.reset()
                    obs = self.env.state().astype(dtype=np.float32)

                if render:
                    self.obs_images.append(self.render(self.env))

            # Per N steps summary update
            if self.stats.n_env_steps % self.stats.n_steps_per_update == 0:
                self.stats.steps_reward_auc += np.trapz([self.stats.steps_reward_last, self.steps_reward])
                self.stats.steps_reward_last = self.steps_reward
                self.stats.epsilon = self.student_agent.epsilon

                self.stats.update_summary_steps(self.steps_reward, self.stats.steps_reward_auc,
                                                self.steps_error_in, self.steps_error_out)

                self.stats.advices_taken = 0.0
                self.stats.exploration_steps_taken = 0
                self.steps_reward = 0.0
                self.steps_error_in = 0.0
                self.steps_error_out = 0.0

            if self.stats.n_env_steps % self.config['evaluation_period'] == 0:
                evaluation_score = self.evaluate()
                print('Evaluation ({}): {}'.format(self.stats.n_episodes, evaluation_score))


            if self.config['save_models'] and self.stats.n_env_steps % self.config['model_save_period'] == 0:
                model_path = os.path.join(os.path.join(self.save_model_path), 'model-{}.ckpt').format(
                    self.stats.n_env_steps)
                print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
                self.saver.save(self.session, model_path)

            if self.stats.n_env_steps >= self.config['n_training_frames']:
                if self.config['save_models']:
                    model_path = os.path.join(os.path.join(self.save_model_path), 'model-{}.ckpt').format(
                        self.stats.n_env_steps)
                    print('[{}] Saving model... {}'.format(self.stats.n_env_steps, model_path))
                    self.saver.save(self.session, model_path)
                break

        print('Env steps: {}'.format(self.stats.n_env_steps))

        if self.config['env_type'] == 0:
            self.save_data()

        self.session.close()

    def write_video(self, images, filename):
        v_w = np.shape(images[0])[0]
        v_h = np.shape(images[0])[1]
        filename_full = os.path.join(self.save_videos_path, str(filename))
        video = cv2.VideoWriter(filename_full + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (v_h, v_w))
        for image in images:
            video.write(image)
        video.release()

    def copy_scripts(self, target_directory):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)
        files = glob.iglob(os.path.join(self.scripts_dir, '*.py'))
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, target_directory)

    def save_config(self, config, filepath):
        fo = open(filepath, "w")
        for k, v in config.items():
            fo.write(str(k) + '>> ' + str(v) + '\n')
        fo.close()

    def evaluate(self):
        eval_render = self.stats.n_evaluations % self.config['evaluation_visualization_period'] == 0

        eval_total_reward = 0.0
        eval_duration = 0

        if self.config['env_type'] == 0 or self.config['env_type'] == 1:
            self.eval_env.set_random_state(self.config['env_evaluation_seed'])

        for i_eval_trial in range(self.config['n_evaluation_trials']):
            eval_obs_images = []

            eval_obs = None
            if self.config['env_type'] == 0:
                eval_obs = self.eval_env.reset()
            elif self.config['env_type'] == 1:
                self.eval_env.reset()
                eval_obs = self.eval_env.state().astype(dtype=np.float32)

            eval_episode_reward = 0.0
            eval_episode_duration = 0

            while True:
                if eval_render:
                    eval_obs_images.append(self.render(self.eval_env))

                eval_action = self.student_agent.greedy_action(eval_obs, evaluation=True)

                eval_obs_next, eval_reward, eval_done = None, None, None
                if self.config['env_type'] == 0:
                    eval_obs_next, eval_reward, eval_done = self.eval_env.step(eval_action)
                elif self.config['env_type'] == 1:
                    eval_reward, eval_done = self.eval_env.act(eval_action)
                    eval_obs_next = self.eval_env.state().astype(dtype=np.float32)

                eval_episode_reward += eval_reward
                eval_duration += 1
                eval_episode_duration += 1
                eval_obs = eval_obs_next

                if eval_done:

                    if self.config['env_type'] == 0:
                        if eval_episode_reward == 1.0:
                            eval_episode_reward = 1.0 - (eval_episode_duration - 24)/76.0

                    if eval_render:
                        eval_obs_images.append(self.render(self.eval_env))
                        self.write_video(eval_obs_images, 'E_{}_{}'.format(str(self.stats.n_episodes),
                                                                           str(self.stats.n_env_steps)))
                        eval_obs_images.clear()
                        eval_render = False
                    eval_total_reward += eval_episode_reward

                    break

        eval_mean_reward = eval_total_reward / float(self.config['n_evaluation_trials'])

        self.stats.evaluation_reward_auc += np.trapz([self.stats.evaluation_reward_last, eval_mean_reward])
        self.stats.evaluation_reward_last = eval_mean_reward

        self.stats.n_evaluations += 1

        self.stats.update_summary_evaluation(eval_mean_reward, eval_duration, self.stats.evaluation_reward_auc)

        return eval_mean_reward

    # ------------------------------------------------------------------------------------------------------------------

    def record_data(self):
        # Grid
        for n in range(len(self.env.passage_positions[0])):
            y = self.env.passage_positions[0][n]
            x = self.env.passage_positions[1][n]
            obs = self.env.generate_obs((y, x))
            #if self.ac:
            if self.action_advising_method == 3:
                self.data_rnd_uncertainty[y, x, self.data_collection_step] = self.student_agent.get_uncertainty(obs)
            if self.rnd_rm is not None:
                self.data_rnd_rm[y, x, self.data_collection_step] = \
                    self.rnd_rm.get_error(obs, normalize=True)

        self.data_rm_obs_counter_all[:, self.data_collection_step] = self.rm_obs_counter_all.copy()
        self.data_rm_obs_counter_teacher[:, self.data_collection_step] = self.rm_obs_counter_teacher.copy()
        self.data_rm_tr_counter_all[:, self.data_collection_step] = self.rm_tr_counter_all.copy()
        self.data_rm_tr_counter_teacher[:, self.data_collection_step] = self.rm_tr_counter_teacher.copy()

        self.data_collection_step += 1

    # ------------------------------------------------------------------------------------------------------------------

    def save_data(self):
        np.save(os.path.join(self.save_data_path, 'RND_Uncertainty.npy'), self.data_rnd_uncertainty)
        np.save(os.path.join(self.save_data_path, 'RND_RM.npy'), self.data_rnd_rm)
        np.save(os.path.join(self.save_data_path, 'RM_Obs_All.npy'), self.data_rm_obs_counter_all)
        np.save(os.path.join(self.save_data_path, 'RM_Obs_Teacher.npy'), self.data_rm_obs_counter_teacher)
        np.save(os.path.join(self.save_data_path, 'RM_TR_All.npy'), self.data_rm_tr_counter_all)
        np.save(os.path.join(self.save_data_path, 'RM_TR_Teacher.npy'), self.data_rm_tr_counter_teacher)