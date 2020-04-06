import argparse
from executor import Executor
import minatar
import lavaworld.environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--process-index', type=int, default=0)
    parser.add_argument('--machine-name', type=str, default='HOME')

    parser.add_argument('--n-training-frames', type=int, default=2000000)
    parser.add_argument('--n-evaluation-trials', type=int, default=20)
    parser.add_argument('--evaluation-period', type=int, default=100)
    parser.add_argument('--evaluation-visualization-period', type=int, default=200)

    # ------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--dqn-gamma', type=float, default=0.99)
    parser.add_argument('--dqn-rm-init', type=int, default=10000)
    parser.add_argument('--dqn-rm-max', type=int, default=100000)
    parser.add_argument('--dqn-target-update', type=int, default=1000)
    parser.add_argument('--dqn-batch-size', type=int, default=32)
    parser.add_argument('--dqn-learning-rate', type=float, default=0.0001)
    parser.add_argument('--dqn-train-per-step', type=int, default=1)
    parser.add_argument('--dqn-train-period', type=int, default=1)
    parser.add_argument('--dqn-adam-epsilon', type=float, default=0.00015)
    parser.add_argument('--dqn-epsilon-start', type=float, default=1.0)
    parser.add_argument('--dqn-epsilon-final', type=float, default=0.1)
    parser.add_argument('--dqn-epsilon-steps', type=int, default=100000)
    parser.add_argument('--dqn-huber-loss-delta', type=float, default=1.0)

    # ------------------------------------------------------------------------------------------------------------------
    # Action-advising related hyperparameters

    # Expert agent to be loaded
    parser.add_argument('--expert-agent-id', type=str, default='')
    parser.add_argument('--expert-agent-seed', type=str, default='')
    parser.add_argument('--expert-agent-checkpoint', type=int, default=0)

    parser.add_argument('--action-advising-rm-th', type=float, default=1.0)
    parser.add_argument('--rm-rnd-learning-rate', type=float, default=0.0001)
    parser.add_argument('--rm-rnd-adam-epsilon', type=float, default=0.00015)

    # ------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--experiment-setup', type=int, default=0)
    parser.add_argument('--use-gpu', action='store_true', default=False)
    parser.add_argument('--generate-plots', action='store_true', default=False)
    parser.add_argument('--save-models', action='store_true', default=False)
    parser.add_argument('--visualization-period', type=int, default=100)
    parser.add_argument('--model-save-period', type=int, default=500)
    parser.add_argument('--env-type', type=int, default=1)
    parser.add_argument('--env-name', type=str, default='asterix')
    parser.add_argument('--env-training-seed', type=int, default=0)
    parser.add_argument('--env-evaluation-seed', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)

    # ------------------------------------------------------------------------------------------------------------------

    config = vars(parser.parse_args())

    env, eval_env = None, None
    if config['env_type'] == 0:  # 0: LavaWorld
        env = lavaworld.environment.Environment(seed=config['env_training_seed'])
        eval_env = lavaworld.environment.Environment(seed=config['env_evaluation_seed'])
    elif config['env_type'] == 1:  # 1: MinAtar
        env = minatar.Environment(config['env_name'],
                                  sticky_action_prob=0.0,
                                  difficulty_ramping=False,
                                  random_seed=config['env_training_seed'],
                                  time_limit=2000)
        eval_env = minatar.Environment(config['env_name'],
                                  sticky_action_prob=0.0,
                                  difficulty_ramping=False,
                                  random_seed=config['env_evaluation_seed'],
                                  time_limit=2000)

    executor = Executor(config, env, eval_env)
    executor.run()

