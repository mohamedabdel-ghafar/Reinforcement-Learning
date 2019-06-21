import tensorflow as tf


class DiscreteQNeuralNetwork:
    def __init__(self, *args):
        raise NotImplementedError

    def run(self, *args):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError


class ImageInputDQNN(DiscreteQNeuralNetwork):
    def __init__(self, num_image_layers, num_channels, num_fc_layers, num_outputs):
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None, num_channels])
        self.last_out = self.state_ph
        for _ in range(num_image_layers):
            self.last_out = tf.keras.layers.Conv2D(5, [5, 5])(self.last_out)
        self.last_out = tf.keras.layers.Flatten()(self.last_out)
        for _ in range(num_fc_layers):
            self.last_out = tf.keras.layers.Dense(125, activation=tf.nn.leaky_relu)(self.last_out)
        self.last_out = tf.keras.layers.Dense(num_outputs, activation=tf.nn.leaky_relu)(self.last_out)

    def run(self):
        return self.last_out

    def get_state_ph(self):
        return self.state_ph


class BasicInputDQNN(DiscreteQNeuralNetwork):
    def __init__(self, state_dim, num_fc_layers, num_actions):
        if isinstance(state_dim, int):
            state_dim = [state_dim]
        assert num_fc_layers > 0
        assert num_actions > 0
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + state_dim)
        self.last_out = self.state_ph
        for _ in range(num_fc_layers//2):
            self.last_out = tf.layers.Dense(100, activation=tf.nn.tanh)(self.last_out)
        for _ in range(num_fc_layers//2):
            self.last_out = tf.layers.Dense(100, activation=tf.nn.relu)(self.last_out)

        self.last_out = tf.layers.Dense(num_actions, activation=tf.identity)(self.last_out)

    def run(self):
        return self.last_out

    def get_state_ph(self):
        return self.state_ph


class BasicInputDuelingDQN(DiscreteQNeuralNetwork):
    def __init__(self, state_dim, num_fc_layers, num_actions):
        if isinstance(state_dim, int):
            state_dim = [state_dim]
        assert  num_actions > 0
        assert num_fc_layers > 0
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None]+state_dim)
        self.last_out = self.state_ph
        for _ in range(num_fc_layers):
            self.last_out = tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu)(self.last_out)
        # advantage channel
        v_s = tf.keras.layers.Dense(1, activation=tf.identity)(self.last_out)
        q_s_a = tf.keras.layers.Dense(num_actions, activation=tf.identity)(self.last_out)
        self.last_out = q_s_a + v_s

    def run(self):
        return self.last_out

    def get_state_ph(self):
        return self.state_ph


class PolicyGradientNN:

    def get_logits_op(self):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError


class BasicInputDiscretePGNN(PolicyGradientNN):
    def __init__(self, state_dim, num_fc_layers, num_actions):
        self.num_actions = num_actions
        if isinstance(state_dim, int):
            state_dim = [state_dim]
            assert num_actions > 0
            assert num_fc_layers > 0
        self.state_ph = tf.placeholder(shape=[None] + state_dim, dtype=tf.float32)
        self.last_out = self.state_ph
        for _ in range(num_fc_layers):
            self.last_out = tf.keras.layers.Dense(100, activation=tf.nn.leaky_relu)(self.last_out)
        self.last_out = tf.keras.layers.Dense(num_actions, activation=tf.identity)(self.last_out)

    def get_logits_op(self):
        return tf.nn.softmax(self.last_out, axis=1)

    def get_state_ph(self):
        return self.state_ph


class ImageInputDiscretePGNN(PolicyGradientNN):
    def __init__(self, num_image_layers, num_channels, num_fc_layers, num_outputs):
        assert num_fc_layers > 0 and num_fc_layers > 0 and num_channels > 0 and num_outputs > 0
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, None, num_channels])
        self.last_out = self.state_ph
        for _ in range(num_image_layers):
            self.last_out = tf.keras.layers.Conv2D(5, [5, 5])(self.last_out)
        self.last_out = tf.keras.layers.Flatten()(self.last_out)
        for _ in range(num_fc_layers):
            self.last_out = tf.keras.layers.Dense(125, activation=tf.nn.leaky_relu)(self.last_out)
        self.last_out = tf.keras.layers.Dense(num_outputs, activation=tf.nn.leaky_relu)(self.last_out)

    def get_logits_op(self):
        return tf.nn.softmax(self.last_out, axis=1)

    def get_state_ph(self):
        return self.state_ph
