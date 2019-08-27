from agents.ddpg import DDPGAgent
from tf_util.nn_helper import SimpleDDPGActor, SimpleDDPGCritic
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym
import numpy as np
import time
from os.path import join, isdir
from os import mkdir, curdir
EXPLORATION_RATIO = 0.1
BATCH_SIZE = 32
GAMMA = 0.01
SAVE_RATE = 100
TRAIN_STEP = 10
EPS_STEP = (1 - EXPLORATION_RATIO) / 10000


def train(env: gym.Env, num_eps=100):
    assert isinstance(env.action_space, gym.spaces.Box)
    prev = np.zeros(shape=env.action_space.shape)
    mu = np.zeros(env.action_space.shape)
    # noise_scale = EXPLORATION_RATIO*abs(env.action_space.low - env.action_space.high) *
    state_dims = env.observation_space.shape
    action_dims = env.action_space.shape
    action_size = 1
    for d in action_dims:
        action_size *= d
    np.random.seed(time.localtime(time.time())[4])
    theta = 0.15
    dt = 1e-2
    sigma = 0.3
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        ddpg_agent = DDPGAgent("sample_agent", SimpleDDPGActor.get_default_ddpg_agent,
                               SimpleDDPGCritic.get_default_ddpg_agent, state_dims, action_size, env.action_space.low,
                               env.action_space.high)
        sess.run(tf.global_variables_initializer())
        train_op = ddpg_agent.get_train_op()
        soft_update_actor = ddpg_agent.actor_soft_update()
        soft_update_critic = ddpg_agent.critic_soft_update()

        def get_action_noise():
            return prev + theta * (mu - prev) * dt + sigma * np.sqrt(dt) * np.random.normal(size=mu.shape)

        s = env.reset()
        print("filling buffer")
        for _ in range(10000):
            # generate random action then store s, a, s',r in replay buffer
            rand_action = np.random.uniform(env.action_space.low, env.action_space.high, action_dims)
            s_p, r, done, _ = env.step(rand_action)
            ddpg_agent.experience(s, rand_action, s_p, r, done)
            if done:
                s = env.reset()
            else:
                s = s_p

        EPS = 1.0
        training_step_counter = 1

        print("training")
        for _ in range(num_eps):
            s = env.reset()
            done = False
            env_step_counter = 0
            curr_episode_rewrads = 0
            while not done:
                take_random = np.random.uniform() <= EPS
                if take_random:
                    action = np.random.uniform(env.action_space.low, env.action_space.high, action_dims)

                else:
                    action = sess.run(ddpg_agent.action_op, feed_dict={
                        ddpg_agent.state_ph: [s]
                    })[0]
                s_p, r, done, _ = env.step(action)
                ddpg_agent.experience(s, action, s_p, r, done)
                env_step_counter += 1
                curr_episode_rewrads += r
                if env_step_counter % TRAIN_STEP == 0:
                    # sample the replay buffer and do a training step
                    states, actions, next_states, rewards, done_arr = ddpg_agent.replay_buffer.sample(BATCH_SIZE)
                    next_states_actions = sess.run(ddpg_agent.action_op, feed_dict={
                        ddpg_agent.state_ph: next_states
                    })
                    q_s_a_n = sess.run(ddpg_agent.q_s_a, feed_dict={
                        ddpg_agent.critic_state_ph: next_states,
                        ddpg_agent.action_ph: next_states_actions
                    })
                    y = np.reshape(rewards, (-1, 1)) + GAMMA * q_s_a_n*done_arr
                    sess.run(train_op, feed_dict={
                        ddpg_agent.y: y,
                        ddpg_agent.state_ph: states,
                        ddpg_agent.action_ph: actions,
                        ddpg_agent.critic_state_ph: states,
                    })
                    sess.run(soft_update_actor)
                    sess.run(soft_update_critic)
                    training_step_counter += 1
                    EPS = max(EPS - EPS_STEP, 0.1)
                    if training_step_counter % SAVE_RATE == 0:
                        if not isdir(join(curdir, "models", "ddpg")):
                            mkdir(join(curdir, "models", "ddpg"))
                        ddpg_agent.save(join(curdir, "models", "ddpg", "model.ckpt"), sess,
                                        training_step_counter)
            print(curr_episode_rewrads)
        ddpg_agent.export(join(curdir, "models", "export",
                               ddpg_agent.name + "__ddpg_SavedModel"), sess)


if __name__ == "__main__":
    train(gym.make("MountainCarContinuous-v0"))

