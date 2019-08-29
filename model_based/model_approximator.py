from tf_util.nn_helper import ModelNN
import tensorflow.compat.v1 as tf


class ModelBasedRL:
    def __init__(self, model_name, model_nn: ModelNN, state_size, action_size):
        with tf.variable_scope(model_name):
            self.state_ph, self.action_ph, self.predict_op = model_nn.get_default(state_size, action_size)
            self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=state_size)
            self.loss = tf.losses.mean_squared_error(self.next_state_ph, self.predict_op)
            optimizer = tf.train.RMSPropOptimizer(lr=0.00025)
            self.train_op = optimizer.minimize(self.loss)

    def get_train_op(self):
        return self.train_op

    def get_action_op(self):
        return self.predict_op

    def export(self, path, sess):
        tf.saved_model.simple_save(session=sess, export_dir=path,
                                   inputs={"state": self.state_ph,
                                           "action": self.state_ph},
                                   outputs={"next_state": self.predict_op})
