from . import RLAgnet, EpisodeBuffer
import tensorflow as tf


# REINFORCE agent
class PolicyGradientAgent(RLAgnet):
    def __init__(self, agent_name, gp_nn, state_dim, num_actions):
        with tf.name_scope(agent_name):
            self.state_ph, self.a_probabilities = gp_nn()
            self.log_p = tf.log(self.a_probabilities)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025)
            self.action_ph = tf.placeholder(dtype=tf.int32, shape=[None])
            self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None])
            one_hot = tf.one_hot(self.action_ph, depth=num_actions)
            p_a = tf.reduce_mean(self.log_p * one_hot)
            self.obj_fn = tf.multiply(self.rew_ph, p_a)
            self.train_op = self.optimizer.minimize(-1 * self.obj_fn)
            self.pd_op = tf.distributions.Categorical(probs=self.a_probabilities)
            self.replay_buffer = EpisodeBuffer(state_dim)
            self.curr_state_buffer = []
            self.action_buffer = []
            self.rew_buffer = []
            self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def clear_local_buffers(self):
        self.curr_state_buffer.clear()
        self.action_buffer.clear()
        self.rew_buffer.clear()

    def get_train_op(self):
        return self.train_op

    def get_action_op(self):
        return self.pd_op.sample()

    def experience(self, state, action, next_state, reward, done):
        self.curr_state_buffer.append(state)
        self.action_buffer.append(action)
        self.rew_buffer.append(reward)
        if done:
            self.replay_buffer.experience(self.curr_state_buffer, self.action_buffer, self.rew_buffer)
            self.clear_local_buffers()

    def save(self, path, sess, step):
        self.saver.save(sess, path, global_step=step)

    def load(self, path, sess):
        pass

    def get_status_ph(self):
        return self.state_ph

    @classmethod
    def agent_type(cls):
        return "Policy_Gradient_Agent"
