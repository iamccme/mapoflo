# coding: utf8

import numpy as np


class NormalActionNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def set_sigma(self, sigma):
        self.sigma = sigma


if __name__ == '__main__':
    pass