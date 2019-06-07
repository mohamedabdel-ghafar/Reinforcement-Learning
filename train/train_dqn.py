import tensorflow as tf
from agents.DQN import DQNAgent
from tf_util.nn_helper import DiscreteQNeuralNetwork, ImageInputDQNN, BasicInputDQNN
import gym
from numpy import random
from os.path import relpath


def train(env: gym.Env):
    UPDATE_FREQ = 100
    EPS = 1
    E_EPS = 0.1
    DOWN_STEP = (EPS - E_EPS) / 1000000
    SAVE_STEP = 10000
    GAMMA = 0.01
    NEW_GEN = 100000
    save_path = relpath("../saves/")
    assert isinstance(env.action_space, gym.spaces.discrete.Discrete)
    num_actions = env.action_space.n
    print(str(num_actions) + " actions available in this env")
    random.seed()
    with tf.Session() as sess:
        def create_nn():
            # image input
            if isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.high.all() == 255 and \
                    env.observation_space.low.all() == 0 and len(env.observation_space.shape) == 3:
                nn = ImageInputDQNN(7, 3, 4, num_actions)
                return nn.state_ph, nn.last_out
            elif isinstance(env.observation_space, gym.spaces.Box):
                assert len(env.observation_space.shape) == 1
                nn = BasicInputDQNN(env.observation_space.shape[0], 5, num_actions)
                return nn.state_ph, nn.last_out
        state_dim = env.observation_space.shape
        dqn_agent = DQNAgent("my_sample_agent", create_nn, state_dim, num_actions, True)

        def e_greedy(state):
            rand = random.rand()
            if rand < EPS:
                return env.action_space.sample()
            else:
                return sess.run(dqn_agent.get_action_op(), feed_dict={dqn_agent.state_ph: [state]})[0]

        total_reward = 0
        reward_per_ep = []
        step_counter = 0
        # warm up
        print("warming up....")
        for j in range(100):
            curr_state = env.reset()
            done = False
            while not done:
                action = e_greedy(curr_state)
                n_state, reward, done, _ = env.step(action)
                dqn_agent.experience(curr_state, action, n_state, reward, done)
                curr_state = n_state
                step_counter += 1
        # initialize all variables
        sess.run(tf.global_variables_initializer())
        for i in range(100000):
            curr_state = env.reset()
            done = False
            reward_per_ep.append(0)
            while not done:
                action = e_greedy(curr_state)
                n_state, reward, done, _ = env.step(action)
                dqn_agent.experience(curr_state, action, n_state, reward, done)
                curr_state = n_state
                total_reward += reward
                reward_per_ep[-1] += reward
                EPS -= DOWN_STEP
                step_counter += 1
                if step_counter % UPDATE_FREQ == 0:
                    states, actions, n_states, rewards, is_terminal = dqn_agent.replay_buffer.sample(32)
                    q_n = sess.run(dqn_agent.score_predict, feed_dict={dqn_agent.state_t_ph: n_states})
                    y = rewards + GAMMA * (1 - is_terminal) * q_n
                    sess.run(dqn_agent.train_op, feed_dict={dqn_agent.state_ph: states, dqn_agent.y: y,
                                                            dqn_agent.action: actions})
                if step_counter % NEW_GEN == 0:
                    sess.run(dqn_agent.get_target_update_op())

                if step_counter % SAVE_STEP == 0:
                    dqn_agent.save(save_path, sess, step_counter)
            print("ep done")
            print("last episode's total rewards: {}".format(reward_per_ep[-1]))


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    train(env)

