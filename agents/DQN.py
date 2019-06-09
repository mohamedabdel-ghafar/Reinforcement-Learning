from . import RLAgnet, ReplayBuffer
import tensorflow as tf
from tensorflow.contrib import slim


class DQNAgent(RLAgnet):
    TAU = 0.01

    def __init__(self, agent_name, create_nn,  state_dim, num_actions, mode: bool):
        self.name = agent_name
        with tf.name_scope(agent_name):
            with tf.name_scope("local"):
                self.state_ph, self.q = create_nn()
                if mode:
                    self.y = tf.placeholder(dtype=tf.float32)
            if mode:
                with tf.name_scope("target"):
                    self.state_t_ph, self.q_ = create_nn()
                    self.t_predict = tf.argmax(self.q_, axis=1)
                    self.score_predict = tf.reduce_mean(tf.multiply(self.q_,
                                                                    tf.one_hot(self.t_predict, depth=num_actions)),
                                                        axis=1)
                self.optimizer = tf.train.AdamOptimizer()
                self.replay_buffer = ReplayBuffer(state_dim)
                self.action = tf.placeholder(dtype=tf.int32)
                self.q_s_a = tf.reduce_mean(tf.multiply(self.q, tf.one_hot(self.action, depth=num_actions)), axis=1)
                self.loss_op = tf.losses.mean_squared_error(self.y, self.q_s_a)
                grads = self.optimizer.compute_gradients(self.loss_op)
                grads, op_vars = zip(*grads)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                grads = zip(grads, op_vars)
                self.train_op = self.optimizer.apply_gradients(grads)
            self.saver = tf.train.Saver(var_list=tf.trainable_variables())

    def get_action_op(self):
        return tf.argmax(self.q, axis=1)

    def experience(self, state, action, next_state, reward, done):
        self.replay_buffer.experience(state, action, next_state, reward, done)

    def get_train_op(self):
        return self.train_op

    def get_target_update_op(self):
        local_vars = tf.trainable_variables(scope=self.name + "/local_q_network")
        target_vars = tf.trainable_variables(scope=self.name + "/target_q_network")
        op = []
        for var, t_var in zip(local_vars, target_vars):
            op.append(tf.assign(t_var, t_var * (1 - self.TAU) + var * self.TAU))
        return op

    def save(self, path, sess, step):
        self.saver.save(sess, path, step)

    def load(self, path, sess):
        ckpt = tf.train.get_checkpoint_state(path)
        model_path = ckpt.model_chekpoint_path
        self.saver.restore(sess, model_path)
