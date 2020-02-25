# coding: utf8

import numpy as np
import hashlib
import torch
import os

from config import *


def process_finished_gc(alphas, gc_winned_imps, demands):
    index = (demands <= gc_winned_imps)
    alphas[index] = -10000.


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


def to_tensor(ndarray, requires_grad=False):
    return torch.tensor(ndarray.astype(np.float32), dtype=torch.float32, requires_grad=requires_grad)


def get_state_id(obs, action):
    time_left = str(round(obs[0, 0], 3))
    target_left = str(round(obs[0, 1], 3))
    pacing_ratio = str(round(obs[0, 2], 3))
    ud_r = str(round(obs[0, 3], 3))
    action = str(round(action, 3))
    key = ','.join([time_left, target_left, pacing_ratio, ud_r, action])
    keymd5 = hashlib.md5()
    keymd5.update(bytes(key, 'utf-8'))

    return keymd5.hexdigest()


def get_latest_reward(reward_dict, batch, agent_index):
    index = 0
    for o, a in zip(batch['obs0'], batch['actions']):
        key = get_state_id(o.reshape(1, -1), a[0])
        assert key in reward_dict
        r = reward_dict[key]
        batch['rewards'][index] = r
        index += 1


def get_pi_obs(state, index):
    if len(state.shape) > 1:
        return state[:, index * PI_FEATURE_NUM: (index + 1) * PI_FEATURE_NUM]
    else:
        return state[index * PI_FEATURE_NUM: (index + 1) * PI_FEATURE_NUM]


def get_state(cur_step, demands, penalties, gc_imps, state, rtb_cpm, gc_pv_ratio, gc_imps_last):
    time_left = np.repeat([1 - float(cur_step + 1) / (MAX_STEP + 1)], MAX_GC_CNT)
    target_left = np.clip((demands - gc_imps) / (demands + EPSILON), 0, 1)

    if gc_imps_last is None:
        pacing_ratio = None
        underdelivery_ratio = None
    else:
        imax = np.iinfo(np.int).max
        p_underdelivery = np.clip(np.clip(demands - gc_imps, 0, imax) - (gc_imps - gc_imps_last) * (MAX_STEP - cur_step + 1), 0, imax)
        underdelivery_ratio = np.log(1 + p_underdelivery / (demands + EPSILON))
        pacing_ratio = np.log(1 + np.clip(target_left / (time_left + EPSILON), 0, np.finfo(float).max))

    return np.array([time_left, target_left, pacing_ratio, underdelivery_ratio]).transpose().reshape(1, -1)


def init():
    os.system("mkdir -p logs")
    os.system("mkdir -p models")


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()
