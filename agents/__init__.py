class RLAgnet:
    def get_action(self, state):
        raise NotImplementedError

    def experience(self, state, action, next_state, reward):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

