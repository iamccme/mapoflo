# coding: utf8

EPSILON = 1E-8

# testing config
IS_EVAL_AND_USE_BEST = True
IS_EVAL_AND_USE_YESTD_BEST = False
IS_TESTING = False
IS_TESTING_FLAG = ""

# data config
PUBID = 1

if PUBID == 1:
    date_gc_cnt = dict([("2019-05-17", 45), ("2019-05-18", 49), ("2019-05-19", 54), ("2019-05-20", 48), ("2019-05-21", 44)])
elif PUBID == 2:
    date_gc_cnt = dict([("2019-05-17", 31), ("2019-05-18", 30), ("2019-05-19", 30), ("2019-05-20", 27), ("2019-05-21", 25)])
else:
    date_gc_cnt = None

TRAIN_DATE_PERIOD = ["2019-05-17"]
GC_CNT = date_gc_cnt[TRAIN_DATE_PERIOD[0]]
DATA_PATH = "data_open_source/"
MAX_GC_CNT = 196
MAX_STEP = 95

# training config
ACTION_SCALE = 0.1
ACTION_NOISE_STDDEV = 0.05
AGENT_ACTION_CNT = 1
ALPHA_INIT_BIAS_STDDEV = 0.05
GAMMA = 1.0
TAU = 0.05
BATCH_SIZE = 32
ACTOR_LR = 1E-5
CRITIC_LR = 1E-2
EPSD_TRAIN_R = 1.
FEATURE_MAX = 20.
ACTION_FREEZING_PROB = 0.1

# model config
MID_LAYER_NODE_NUM = 100
PI_FEATURE_NUM = 4
CRITIC_FEATURE_NUM = PI_FEATURE_NUM
CRITIC_ACTION_NUM = 1

# memory config
MEMORY_CAPACITY = 10000
MEMORY_ALPHA = 0.5  # [0~1] convert the importance of TD error to priority
MEMORY_BETA = 0.4  # importance-sampling, from initial value increasing to 1
MEMORY_BETA_INC_RATE = 1E-4
MEMORY_MIN_SIZE = MEMORY_CAPACITY * 0.1
MEMORY_ACTION_CNT = 1
MEMORY_CRITIC_FEATURE_NUM = PI_FEATURE_NUM
MEMORY_REWARD_CNT = 1

# prioritized sweeping
PS_ALPHA = 0.8
PS_BETA = 0.6
WEIGHTED_IS_FLAG = False
