from numpy import random, array, reshape, append, ndarray, concatenate


class RLAgnet:
    def get_action_op(self):
        raise NotImplementedError

    def experience(self, state, action, next_state, reward, done):
        raise NotImplementedError

    def get_train_op(self):
        raise NotImplementedError

    def save(self, path, sess, step):
        raise NotImplementedError

    def load(self, path, sess):
        raise NotImplementedError


class ReplayBuffer:
    max_len = 100000

    def __init__(self, state_dim):
        self.buffer = None
        self.state_dim = state_dim
        self.state_size = 1
        for element in self.state_dim:
            self.state_size *= element
        self.next = 0

        def preprocess_element(state, action, next_state, reward, done):
            element = concatenate((reshape(state, (-1)), [action], reshape(next_state, (-1)), [reward, done]),
                                  axis=0)
            return reshape(element, (-1))
        self.preprocess = preprocess_element

    def experience(self, state, action, next_state, reward, done):
        n_el = self.preprocess(state, action, next_state, reward, done)
        if self.buffer is None:
            self.buffer = reshape(n_el, (1, -1))
        elif self.next < len(self.buffer):
            self.buffer[self.next] = n_el
        else:
            self.buffer = append(self.buffer, reshape(n_el, [1, -1]), axis=0)
        self.next = (self.next + 1) % self.max_len

    def sample(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            indx = random.randint(len(self.buffer))
            batch.append(self.buffer[indx])
        batch = array(batch)
        states = reshape(batch[:, range(self.state_size)], (-1, )+self.state_dim)
        actions = batch[:, self.state_size]
        n_states = reshape(batch[:, range(self.state_size+1, self.state_size*2+1)], (-1, )+self.state_dim)
        rewards = batch[:, 2*self.state_size+1]
        done = batch[:, 2*self.state_size+2]
        return states, actions, n_states, rewards, done


class PrioritizedReplayBuffer():
    def __init__(self):
        pass

    def experience(self, state, action, next_state, reward, done, priority):
        pass

    def sample(self, batch_size):
        pass
