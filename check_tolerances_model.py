import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

num_sims = 10
num_sims_each = 500000
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
        self.lengths = self.init_chain()
        self.state = RightState(self.lengths, 0)

    def reset(self):
        self.tot_reward = 0
        self.transition_count = 0
        self.state = RightState(self.lengths, 0)

    @staticmethod
    def init_chain():
        lengths = noisy_lengths(orig_mid_bounds, tol)
        return lengths

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
    is_reward_arr = np.empty(num_sims_each, dtype=bool)
    tot_reward_arr = np.empty(num_sims_each)
    transition_count_arr = np.empty(num_sims_each)
    for i in range(num_sims_each):
        mc.reset()
        is_done = False
        while not is_done:
            is_done, is_reward, tot_reward, transition_count = mc.transition()
        is_reward_arr[i], tot_reward_arr[i], transition_count_arr[i] = is_reward, tot_reward, transition_count
    is_reward_mean = np.mean(is_reward_arr)
    is_reward_std = np.std(is_reward_arr)
    tot_reward_arr = tot_reward_arr[is_reward_arr]
    tot_reward_mean = np.mean(tot_reward_arr)
    tot_reward_std = np.std(tot_reward_arr)
    transition_count_arr = transition_count_arr[is_reward_arr]
    transition_count_mean = np.mean(transition_count_arr)
    transition_count_std = np.std(transition_count_arr)
    return is_reward_mean, is_reward_std, tot_reward_mean, tot_reward_std, transition_count_mean, transition_count_std


if __name__ == "__main__":
    is_reward_mean_arr = np.empty(num_sims)
    is_reward_std_arr = np.empty(num_sims)
    tot_reward_mean_arr = np.empty(num_sims)
    tot_reward_std_arr = np.empty(num_sims)
    transition_count_mean_arr = np.empty(num_sims)
    transition_count_std_arr = np.empty(num_sims)
    for i in range(num_sims):
        is_reward_mean_arr[i], is_reward_std_arr[i], tot_reward_mean_arr[i], tot_reward_std_arr[i], \
        transition_count_mean_arr[i], transition_count_std_arr[i] = run_simulation()
    fig, axes = plt.subplots(3, 1)
    axes[0].errorbar(np.arange(num_sims), is_reward_mean_arr, yerr=is_reward_std_arr/np.sqrt(num_sims_each))
    axes[1].errorbar(np.arange(num_sims), tot_reward_mean_arr, yerr=tot_reward_std_arr/np.sqrt(num_sims_each))
    axes[2].errorbar(np.arange(num_sims), transition_count_mean_arr, yerr=transition_count_std_arr/np.sqrt(num_sims_each))
    for ax in axes:
        ax.grid()
    plt.show()

