import numpy as np

np.random.seed(42)

num_sims = 10000
n = 10
L = 1
R = 0.5
T = 1 - R
tol = 0.05

orig_mid_bounds = np.sort(np.random.random(n - 1)) * L


def lengths_from_mid_bounds(mid_bounds):
    bounds_of_length = np.append(0, np.append(mid_bounds, L))
    return np.diff(bounds_of_length)


def noisy_lengths(mid_bounds, tol):
    new_bounds = [-1]
    while new_bounds[0] <= 0 or new_bounds[-1] >= L:
        new_bounds = np.sort(mid_bounds + np.random.randn(np.size(mid_bounds)) * tol)
    return lengths_from_mid_bounds(new_bounds)


class MarkovChain:
    def __init__(self):
        self.tot_reward = 0
        self.transition_count = 0
        self.lengths, self.state = self.init_chain()

    @staticmethod
    def init_chain():
        lengths = noisy_lengths(orig_mid_bounds, tol)
        starting_state = RightState(lengths, 0)
        return lengths, starting_state

    def transition(self):
        self.tot_reward += self.state.reward
        self.state = self.state.transition()
        self.transition_count += 1
        if isinstance(self.state, DeathState):
            is_done = True
            is_reward = False
        elif isinstance(self.state, EndState):
            is_done = True
            is_reward = True
        else:
            is_done = False
            is_reward = False
        return is_done, is_reward, self.tot_reward, self.transition_count


class State:
    def __init__(self):
        self.reward = 0

    @property
    def reward(self):
        return self._reward

    @reward.setter
    def reward(self, value):
        self._reward = value

    def transition(self):
        raise NotImplementedError()


class RightState(State):
    def __init__(self, lengths, ind):
        super().__init__()
        self.lengths = lengths
        self.reward = lengths[ind]
        self.ind = ind

    def transition(self):
        is_R = np.random.random() < R
        if is_R:
            return LeftState(self.lengths, self.ind)
        else:
            if self.ind < n - 1:
                return RightState(self.lengths, self.ind + 1)
            else:
                return EndState()


class LeftState(State):
    def __init__(self, lengths, ind):
        super().__init__()
        self.lengths = lengths
        self.reward = lengths[ind]
        self.ind = ind

    def transition(self):
        is_R = np.random.random() < R
        if is_R:
            return RightState(self.lengths, self.ind)
        else:
            if self.ind > 0:
                return LeftState(self.lengths, self.ind - 1)
            else:
                return DeathState()


class DeathState:
    pass


class EndState:
    pass


def run_simulation():
    mc = MarkovChain()
    is_done = False
    while not is_done:
        is_done, is_reward, tot_reward, transition_count = mc.transition()
    return is_reward, tot_reward, transition_count


if __name__ == "__main__":
    is_reward_arr = np.empty(num_sims)
    tot_reward_arr = np.empty(num_sims)
    transition_count_arr = np.empty(num_sims)
    for i in range(num_sims):
        is_reward_arr[i], tot_reward_arr[i], transition_count_arr[i] = run_simulation()
    is_reward_mean = np.mean(is_reward_arr)
    print(is_reward_mean)

# TODO need to calculate expectancy over same chain, so do each one a couple of times without changing weights
# TODO add analysis of rewards
