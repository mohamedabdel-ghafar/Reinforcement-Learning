from . import RLAgnet, ReplayBuffer
import tensorflow as tf

# my simplified implementation of DDPG algorithm, the algorithm need continuous action space


class DDPGAgent(RLAgnet):
    TAU = 0.01

    def __init__(self, agent_name, create_actor_nn, create_critic_nn, state_dim, action_size):
        with tf.name_scope(agent_name):
            self.y = tf.placeholder(dtype=tf.float32, shape=[None]+action_size)
            self.name = agent_name
            with tf.name_scope("local"):
                with tf.name_scope("actor"):
                    self.state_ph, self.action_op = create_actor_nn(state_dim, action_size)
                with tf.name_scope("critic"):
                    self.critic_state_ph, self.action_ph, self.q_s_a = create_critic_nn(state_dim, action_size)
            with tf.name_scope("target"):
                with tf.name_scope("actor"):
                    self.t_state_ph, self.t_action_op = create_actor_nn(state_dim, action_size)
                with tf.name_scope("critic"):
                    self.t_critic_state_ph, self.t_action_ph, self.t_q_s_a = create_critic_nn(state_dim, action_size)
            self.optimizer = tf.train.RMSPropOptimizer(0.00025)
            loss = tf.losses.mean_squared_error(self.y, self.action_op)
            loss_to_action_grads, _ = zip(*self.optimizer.compute_gradients(loss, self.action_ph))
            loss_to_action_grads = tf.gradients(self.action_op, tf.trainable_variables(self.name + "/local/actor"),
                                                loss_to_action_grads)
            loss_to_action_grads = zip(loss_to_action_grads, tf.trainable_variables(self.name + "/local/actor"))
            self.train_actor_op = self.optimizer.apply_gradients(loss_to_action_grads)
            self.train_critic_op = self.optimizer.minimize(loss)
        self.saver = tf.train.Saver()
        self.replay_buffer = ReplayBuffer(state_dim)

    def experience(self, state, action, next_state, reward, done):
        self.replay_buffer.experience(state, action, next_state, reward, done)

    def critic_soft_update(self):
        op = []
        for v, t_v in zip(tf.trainable_variables(self.name + "/local/critic"),
                          tf.trainable_variables(self.name + "/target/critic")):
            op.append(tf.assign(t_v, self.TAU * v + (1 - self.TAU) * t_v))
        return op

    def actor_soft_update(self):
        op = []
        for v, t_v in zip(tf.trainable_variables(self.name + "/local/actor"),
                          tf.trainable_variables(self.name + "/target/actor")):
            op.append(tf.assign(t_v, self.TAU * v + (1 - self.TAU) * t_v))
        return op

    def get_status_ph(self):
        return self.state_ph

    def get_action_op(self):
        return self.action_op

    def get_train_op(self):
        return [self.train_critic_op, self.train_actor_op]

    def save(self, path, sess, step):
        self.saver.save(sess, path, step)

    def load(self, path, sess):
        self.saver.restore(sess, path)

    @classmethod
    def agent_type(cls):
        return "DEEP_DETERMINISTIC_POLICY_GRADIENT_AGENT"
