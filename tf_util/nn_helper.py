import tensorflow.compat.v1 as tf


class DiscreteQNeuralNetwork:
    def __init__(self, *args):
        raise NotImplementedError

    def run(self, *args):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError


# for model-based rl
class ModelNN:
    def __init__(self, state_dim, action_dim):
        raise NotImplementedError

    def get_predict_op(self):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError

    def get_action_ph(self):
        raise NotImplementedError

    @classmethod
    def get_default(cls, state_dim, action_dim):
        nn = cls(state_dim, action_dim)
        return nn.get_state_ph(), nn.get_action_ph(), nn.get_predict_op()


class BasicModelNN(ModelNN):
    def __init__(self, state_dim, action_dim):
        assert(isinstance(state_dim, int) or isinstance(state_dim, list)
               or isinstance(state_dim, tuple))
        assert(isinstance(action_dim, int) or isinstance(action_dim, list)
               or isinstance(action_dim, tuple))
        assert isinstance(state_dim, int) or len(state_dim) < 3
        self.state_size = 1
        self.action_size = 1
        action_dim = list(action_dim)
        state_dim = list(state_dim)
        for d in state_dim:
            self.state_size *= d
        for d in action_dim:
            self.action_size *= d
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size])
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_size])
        self.predict_op = tf.concat([self.state_ph, self.action_ph], axis=1)
        n_units = self.state_size + self.action_size // 2
        n_units = max(n_units, 1)
        for _ in range(3):
            self.predict_op = tf.keras.layers.Dense(n_units, activation=tf.nn.leaky_relu)(self.predict_op)
            n_units = n_units // 2
            n_units = max(n_units, 1)
        self.predict_op = tf.keras.layers.Dense(self.state_size)

    def get_action_ph(self):
        return self.action_ph

    def get_state_ph(self):
        return self.state_ph

    def get_predict_op(self):
        return self.predict_op


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


class DDPGActorNetwork:

    def __init__(self, state_dims, action_dims, action_lows, action_highs):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError

    def get_action_op(self):
        raise NotImplementedError

    @classmethod
    def get_default_ddpg_agent(cls, state_dims, num_actions, action_lows, action_highs):
        agent = cls(state_dims, num_actions, action_lows, action_highs)
        return agent.get_state_ph(), agent.get_action_op()


class DDPGCriticNetwork:
    def __init__(self, state_dims, action_dims):
        raise NotImplementedError

    def get_state_ph(self):
        raise NotImplementedError

    def get_action_ph(self):
        raise NotImplementedError

    def get_value_op(self):
        raise NotImplementedError

    @classmethod
    def get_default_ddpg_agent(cls, state_dims, num_actions):
        agent = cls(state_dims, num_actions)
        return agent.get_state_ph(), agent.get_action_ph(), agent.get_value_op()


class SimpleDDPGActor(DDPGActorNetwork):
    def __init__(self, state_dims, num_actions, action_lows, action_highs):
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(state_dims))
        self.action_op = self.state_ph
        for _ in range(3):
            self.action_op = tf.layers.Dense(100, activation=tf.nn.relu)(self.action_op)
            self.action_op = tf.layers.BatchNormalization()(self.action_op)
        self.action_op = tf.layers.Dense(num_actions, activation=tf.nn.sigmoid)(self.action_op)
        scale = action_highs - action_lows
        self.action_op = tf.multiply(self.action_op, scale)
        self.action_op = action_lows + self.action_op

    def get_action_op(self):
        return self.action_op

    def get_state_ph(self):
        return self.state_ph


class SimpleDDPGCritic(DDPGCriticNetwork):
    def __init__(self, state_dims, action_size):
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None]+list(state_dims))
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
        l_1 = tf.layers.Dense(100)(self.state_ph)
        l_2 = tf.layers.Dense(100)(self.action_ph)
        num_units = 100
        self.q_s_a = l_1 + l_2
        for _ in range(2):
            self.q_s_a = tf.layers.Dense(num_units)(self.q_s_a)
            num_units //= 2
            self.q_s_a = tf.layers.BatchNormalization()(self.q_s_a)
        self.q_s_a = tf.layers.Dense(1)(self.q_s_a)

    def get_state_ph(self):
        return self.state_ph

    def get_action_ph(self):
        return self.action_ph

    def get_value_op(self):
        return self.q_s_a




