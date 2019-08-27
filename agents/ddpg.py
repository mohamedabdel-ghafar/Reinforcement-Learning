from . import RLAgnet, ReplayBuffer
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# my simplified implementation of DDPG algorithm, the algorithm needs continuous action space


class DDPGAgent(RLAgnet):
    TAU = 0.01

    def __init__(self, agent_name, create_actor_nn, create_critic_nn, state_dim, action_size, action_l, action_h):
        with tf.variable_scope(agent_name):
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            self.name = agent_name
            with tf.variable_scope("local"):
                with tf.variable_scope("actor"):
                    self.state_ph, self.action_op = create_actor_nn(state_dim, action_size, action_l, action_h)
                with tf.variable_scope("critic"):
                    self.critic_state_ph, self.action_ph, self.q_s_a = create_critic_nn(state_dim, action_size)
            with tf.variable_scope("target"):
                with tf.variable_scope("actor"):
                    self.t_state_ph, self.t_action_op = create_actor_nn(state_dim, action_size, action_l, action_h)
                with tf.variable_scope("critic"):
                    self.t_critic_state_ph, self.t_action_ph, self.t_q_s_a = create_critic_nn(state_dim, action_size)
            self.optimizer = tf.train.RMSPropOptimizer(0.00025)
            loss = tf.losses.mean_squared_error(self.y, self.q_s_a)
            trainable_vars = tf.trainable_variables(scope=self.name+"/local/actor")
            loss_to_action_grads, _ = zip(*self.optimizer.compute_gradients(loss, self.action_ph))
            loss_to_action_grads = tf.gradients(ys=self.action_op, xs=trainable_vars, grad_ys=loss_to_action_grads)
            loss_to_action_grads = zip(loss_to_action_grads, trainable_vars)
            self.train_actor_op = self.optimizer.apply_gradients(loss_to_action_grads)
            self.train_critic_op = self.optimizer.minimize(loss)
        self.saver = tf.train.Saver()
        self.replay_buffer = ReplayBuffer(state_dim)

    def experience(self, state, action, next_state, reward, done):
        self.replay_buffer.experience(state, action, next_state, reward, done)

    def critic_soft_update(self):
        op = []
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+"/local/critic")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + "/target/critic")
        assert len(local_vars) == len(target_vars)
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

    def export(self, path, sess):
        tf.saved_model.simple_save(session=sess, export_dir=path,
                                   inputs={"state": self.state_ph},
                                   outputs={"action": self.action_op})
