# coding: utf8

from collections import defaultdict
from torch.optim import Adam
from nn_def import (Actor, RewardNet)
from memory import Memory
from utils import *
from config import *


class Model(object):
    def __init__(self, id, env, action_noise=None, action_bounds=(-1., 1.)):
        self.reward_dict = defaultdict(float)
        self.id = id
        self.env = env
        self.pi_fn = PI_FEATURE_NUM
        self.critic_fn = CRITIC_FEATURE_NUM
        self.action_bounds = action_bounds

        self.actor = Actor(self.pi_fn, AGENT_ACTION_CNT)
        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)

        self.reward_net = RewardNet(self.critic_fn, CRITIC_ACTION_NUM)
        self.rn_optim = Adam(self.reward_net.parameters(), lr=CRITIC_LR)

        self.memory = Memory()
        self.action_noise = action_noise

    def pi(self, state, all_memory_ready, apply_noise=True, done=False):
        if done:
            return np.array([0.])

        if not all_memory_ready and apply_noise:
            sigma = np.clip(self.memory.size() / float(MEMORY_MIN_SIZE), 0, 1) * ACTION_NOISE_STDDEV
            self.action_noise.set_sigma(sigma)
            return np.clip(np.array([0.]) + self.action_noise(), self.action_bounds[0], self.action_bounds[1])

        with torch.no_grad():
            obs = get_pi_obs(state, self.id)
            action = float(self.actor(to_tensor(obs)).detach().numpy())

            if self.action_noise is not None and apply_noise:
                noise = self.action_noise()
                action += noise

            action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

            return action

    def is_memory_ready(self):
        return self.memory.size() > MEMORY_MIN_SIZE

    def get_joint_info(self, batch, cur_date):
        interval = int(batch.shape[1] / MAX_GC_CNT)
        index = list(self.env.date_gc_index[cur_date])
        index.sort()
        index = np.array(index)
        if interval == 1:
            return batch[:, index]
        else:
            all_index = index
            for i in range(len(batch) - 1):
                all_index = np.concatenate([all_index, index + (i + 1) * MAX_GC_CNT])

            return np.reshape(np.reshape(batch, (-1, interval))[all_index], (-1, interval * len(index)))

    def train(self, all_agents, cur_date):
        if self.memory.size() < BATCH_SIZE * 50:
            return None

        # Get a batch.
        idx, batch, isw = self.memory.sample(batch_size=BATCH_SIZE)

        # Get latest reward, since the reward is updating during iteration
        get_latest_reward(self.reward_dict, batch, self.id)

        self.rn_optim.zero_grad()
        s = to_tensor(batch['obs0'], requires_grad=True)
        a = to_tensor(batch['actions'], requires_grad=True)
        r = to_tensor(batch['rewards'], requires_grad=True)
        q = self.reward_net([s, a + ACTION_SCALE])
        q_loss = torch.nn.MSELoss()(q, r)
        q_loss.backward()
        self.rn_optim.step()

        tderr = np.abs((q - r).detach().numpy())
        self.memory.batch_update(idx, tderr)

        self.actor_optim.zero_grad()
        s = to_tensor(batch['obs0'], requires_grad=True)
        step_left = s[:, 0] * (MAX_STEP + 1)
        a_loss = -(self.reward_net([s, self.actor(s) + ACTION_SCALE]) * step_left).mean()
        a_loss.backward()
        self.actor_optim.step()

        return q_loss.detach().numpy(), a_loss.detach().numpy()
