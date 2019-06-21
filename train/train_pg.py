from agents.pg import PolicyGradientAgent
import tensorflow as tf
from tf_util.nn_helper import BasicInputDiscretePGNN, ImageInputDiscretePGNN
import gym
from tf_util.deployer import deploy_model


def train_discrete(env: gym.Env, gym_id):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    num_actions = env.action_space.n
    print("{} environment has {} actions".format(gym_id, num_actions))

    def create_nn():
        if isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.high.all() == 255 and \
                env.observation_space.low.all() == 0 and len(env.observation_space.shape) == 3:
            nn = ImageInputDiscretePGNN(7, 3, 4, num_actions)
            return nn.state_ph, nn.get_logits_op()
        elif isinstance(env.observation_space, gym.spaces.Box):
            assert len(env.observation_space.shape) == 1
            nn = BasicInputDiscretePGNN(env.observation_space.shape[0], 2, num_actions)
            return nn.state_ph, nn.get_logits_op()

    with tf.Session() as sess:
        pg_agent = PolicyGradientAgent("my_sample_agent", create_nn, env.observation_space.shape, num_actions)
        sess.run(tf.global_variables_initializer())
        rew_per_ep = []
        for _ in range(1000):
            # collect some episodes
            rew_per_ep.append(0)
            for _ in range(10):
                print("collecting")
                curr_state = env.reset()
                pg_agent.clear_local_buffers()
                done = False
                while not done:
                    action = sess.run(pg_agent.get_action_op(), feed_dict={pg_agent.state_ph: [curr_state]})[0]
                    n_state, reward, done, _ = env.step(action)
                    rew_per_ep[-1] += reward
                    pg_agent.experience(curr_state, action, n_state, reward, done)
                    curr_state = n_state
            print("avg reward per ep {}".format(sum(rew_per_ep)/len(rew_per_ep)))
            # train using collected episodes
            for _ in range(7):
                print("training")
                flat_states, actions, rewards = pg_agent.replay_buffer.single_sample()
                sess.run(pg_agent.get_train_op(), feed_dict={pg_agent.state_ph: flat_states,
                                                             pg_agent.action_ph: actions,
                                                             pg_agent.rew_ph: rewards})

    # deploy_model(pg_agent, sess, "../saves", gym_id, "v_1")


if __name__ == "__main__":
    env_id = "MountainCar-v0"
    env = gym.make(env_id)
    train_discrete(env, env_id)

