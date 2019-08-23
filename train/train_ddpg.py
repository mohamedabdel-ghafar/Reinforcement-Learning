from agents.ddpg import DDPGAgent
from tf_util.nn_helper import SimpleDDPGActor, SimpleDDPGCritic
import tensorflow as tf
import gym
import numpy as np
import time
EXPLORATION_RATIO = 0.1


def train(env: gym.Env, gym_id):
    assert isinstance(env.action_space, gym.spaces.Box)
    prev = np.zeros(shape=env.action_space.shape)
    mu = np.zeros(env.action_space.shape)
    # noise_scale = EXPLORATION_RATIO*abs(env.action_space.low - env.action_space.high) *
    state_dims = env.observation_space.shape
    action_dims = env.observation_space.shape

    np.random.seed(time.localtime(time.time())[4])
    theta = 0.15
    dt = 1e-2
    sigma = 0.3
    ddpg_agent = DDPGAgent("sample_agent", SimpleDDPGActor.get_default_ddpg_agent,
                           SimpleDDPGCritic.get_default_ddpg_agent, state_dims, action_dims)



    def get_action_noise():
        return prev + theta * (mu - prev) * dt + sigma * np.sqrt(dt) * np.random.normal(size=mu.shape)


if __name__ == "__main__":
    train(gym.make("MountainCarContinuous-v0"), "MountainCarContinuous-v0")

