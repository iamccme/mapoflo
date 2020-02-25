# coding: utf8

from environment import Environment
from mapoflo import MAPOFLO
from utils import *

init()
env = Environment()
mapoflo = MAPOFLO(env)
mapoflo.run()
