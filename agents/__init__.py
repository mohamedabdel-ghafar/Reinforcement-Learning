from numpy import random, array, reshape, append, cumsum, concatenate, hstack


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

    def get_status_ph(self):
        raise NotImplementedError

    @classmethod
    def agent_type(cls):
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


class PrioritizedReplayBuffer:
    def __init__(self, state_dim):
        pass

    def experience(self, state, action, next_state, reward, done, priority):
        pass

    def sample(self, batch_size):
        pass


class EpisodeBuffer:
    max_len = 2000
    # max len here stands for a number of episodes not state transitions so we have to reduce it
    # note: only support discrete action space for now

    def __init__(self, state_dim):
        self._buffer = []
        self._next = 0
        self._state_dim = state_dim
        if isinstance(state_dim, int):
            self._state_dim = [state_dim]
        self.state_size = 1
        for x in self._state_dim:
            self.state_size *= x

    def experience(self, state_l, action_l:list, rew_l):
        rew_per_state = reshape(list(reversed(cumsum(list(reversed(rew_l)), axis=0))), (-1))
        state_l = reshape(state_l, (-1, self.state_size))
        action_l = array(action_l)
        # rew_l = array(rew_l)
        n_el = (state_l, action_l, rew_per_state)
        if self._next < len(self._buffer):
            self._buffer[self._next] = n_el
        else:
            self._buffer.append(n_el)
        self._next = (1 + self._next) % self.max_len

    def sample(self, batch_size):
        ret = []
        while len(ret) < batch_size:
            indx = random.randint(len(self._buffer))
            ret.append(self._buffer[indx])
        return ret

    def single_sample(self):
        indx = random.randint(len(self._buffer))
        return self._buffer[indx]
