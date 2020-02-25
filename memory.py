# coding: utf8

from sum_tree import SumTree
from utils import *
from collections import namedtuple


class Memory(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    beta = MEMORY_BETA

    def __init__(self):
        self.limit = MEMORY_CAPACITY
        self.err_tree = SumTree(MEMORY_CAPACITY)
        self.action_shape = (0, MEMORY_ACTION_CNT)
        self.reward_shape = (0, MEMORY_REWARD_CNT)
        self.terminal_shape = self.action_shape
        self.observation_shape = (0, MEMORY_CRITIC_FEATURE_NUM)
        self.store_times = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal'))

    def size(self):
        return self.limit if self.store_times > self.limit else self.store_times

    def sample(self, batch_size):
        idxes = np.empty(self.reward_shape, dtype=np.int32)
        isw = np.empty(self.reward_shape, dtype=np.float32)
        obs0 = np.empty(self.observation_shape, dtype=np.float32)
        obs1 = np.empty(self.observation_shape, dtype=np.float32)
        actions = np.empty(self.action_shape, dtype=np.float32)
        rewards = np.empty(self.reward_shape, dtype=np.float32)
        terminals = np.empty(self.terminal_shape, dtype=np.bool)
        nan_state = np.array([np.nan] * self.observation_shape[1])

        self.beta = np.min([1., self.beta + MEMORY_BETA_INC_RATE])  # max = 1
        max_td_err = np.max(self.err_tree.tree[-self.err_tree.capacity:])
        idx_set = set()
        for i in range(batch_size * 2):  # sample maximum batch_size * 2 times to get batch_size different instances
            v = np.random.uniform(0, self.err_tree.total_p)
            idx, td_err, trans = self.err_tree.get_leaf(v)
            if batch_size == len(idx_set):
                break
            if idx not in idx_set:
                idx_set.add(idx)
            else:
                continue
            if (trans.state == 0).all():
                continue
            idxes = np.row_stack((idxes, np.array([idx])))
            isw = np.row_stack((isw, np.array([np.power(self._getPriority(td_err) / max_td_err, -self.beta)])))
            obs0 = np.row_stack((obs0, trans.state))
            obs1 = np.row_stack((obs1, nan_state if trans.terminal.all() else trans.next_state))
            actions = np.row_stack((actions, trans.action))
            rewards = np.row_stack((rewards, trans.reward))
            terminals = np.row_stack((terminals, trans.terminal))

        result = {
            'obs0': array_min2d(obs0),
            'actions': array_min2d(actions),
            'rewards': array_min2d(rewards),
            'obs1': array_min2d(obs1),
            'terminals': array_min2d(terminals),
        }

        return idxes, result, isw

    def _getPriority(self, error):
        return (error + EPSILON) ** MEMORY_ALPHA

    def append(self, obs0, action, reward, obs1, terminal, err, training=True):
        if not training:
            return
        trans = self.Transition(obs0, action, reward, obs1, terminal)
        self.err_tree.add(self._getPriority(err), trans)
        self.store_times += 1

    def batch_update(self, tree_idx, errs):
        errs = np.abs(errs) + EPSILON  # convert to abs and avoid 0
        ps = np.power(errs, MEMORY_ALPHA)
        for ti, p in zip(tree_idx, ps):
            self.err_tree.update(ti, p[0])

    @property
    def nb_entries(self):
        return self.store_times
