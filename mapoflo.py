# coding: utf8

from utils import *
from noise import *
from model import Model

import datetime
import copy
import os
import sys


class MAPOFLO(object):
    def __init__(self, env):
        # logging config
        if not (IS_TESTING or IS_EVAL_AND_USE_BEST or IS_EVAL_AND_USE_YESTD_BEST):
            time = datetime.datetime.now().strftime("%m%d%H%M")
            model_name = "mapoflo"
            filename = ".".join([time, TRAIN_DATE_PERIOD[0], str(PUBID), model_name])
            f = open('logs/' + filename + '.log.txt', 'w')
            sys.stdout = Tee(sys.stdout, f)
        print("IS_EVAL_AND_USE_BEST", IS_EVAL_AND_USE_BEST, "IS_EVAL_AND_USE_YESTD_BEST", IS_EVAL_AND_USE_YESTD_BEST, "IS_TESTING", IS_TESTING)

        self.env = env
        self.cur_date = None
        self.is_memory_ready = False
        self.gc_bidders = []
        self.action_noise = NormalActionNoise(mu=0, sigma=ACTION_NOISE_STDDEV)

        if IS_TESTING:
            print("Loading model", IS_TESTING_FLAG)

        for id in range(MAX_GC_CNT):
            agent = Model(id=id, env=env, action_bounds=(-ACTION_SCALE, ACTION_SCALE), action_noise=self.action_noise)
            if IS_TESTING:
                model_date = (datetime.datetime.strptime(TRAIN_DATE_PERIOD[0], "%Y-%m-%d") - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                agent.actor = torch.load("models/{}/{:s}/{:d}".format(str(PUBID) + "_" + model_date, IS_TESTING_FLAG, id))
            self.gc_bidders.append(agent)

    def update_alpha(self, state, alphas, penalties, dones, apply_noise=False):
        actions = []
        for i in range(len(self.gc_bidders)):
            if i not in self.env.date_gc_index[self.cur_date]:
                actions.append(0.)
                continue

            agent = self.gc_bidders[i]
            action = agent.pi(state, self.is_memory_ready, apply_noise=apply_noise, done=dones[0, i])
            actions.append(float(action))
            if dones[0, i]:
                lb = -10000
            else:
                lb = -20
            alphas[i] = np.clip(alphas[i] + float(action) * penalties[i], lb, penalties[i])

        return actions

    def memory_store(self, s, a, r, ns, dones, done_flag, step, alphas):
        memory_ready_list = []
        for i in range(len(self.gc_bidders)):
            if i in self.env.date_gc_index[self.cur_date]:
                memory_ready_list.append(self.gc_bidders[i].memory.size() > MEMORY_MIN_SIZE)

            if i not in self.env.date_gc_index[self.cur_date] or done_flag[i]:
                continue

            o = get_pi_obs(s, i)
            no = get_pi_obs(ns, i)
            a_ = np.array([a[0, i]]).reshape(-1, 1)
            d = np.array([dones[0, i]]).reshape(-1, 1)
            key = get_state_id(o, a_[0, 0])
            if self.gc_bidders[i].reward_dict[key] < r:
                self.gc_bidders[i].reward_dict[key] = r

            self.gc_bidders[i].memory.append(o, a_, np.array([self.gc_bidders[i].reward_dict[key]]).reshape(-1, 1), no, d, 0.1)
            if d[0, 0]:
                done_flag[i] = True

        if np.array(memory_ready_list).all():
            self.is_memory_ready = True

    def store_traj(self, traj, outcome, opt_outcome):
        done_flag = np.repeat([False], MAX_GC_CNT)
        for i in range(len(traj)):
            s, a, r, ns, dones, alphas = traj[i]
            r = 10 ** (outcome / opt_outcome)

            self.memory_store(s, a, r, ns, dones, done_flag, i, alphas)

    def train(self):
        loss = []
        for i in range(len(self.gc_bidders)):
            if i not in self.env.date_gc_index[self.cur_date]:
                continue
            rt = self.gc_bidders[i].train(self.gc_bidders, self.cur_date)
            if rt is None:
                continue
            q_loss, a_loss = rt
            loss.append((float(q_loss), float(a_loss)))

        return np.mean(np.array(loss), axis=0) if loss else loss

    def allocation(self, cur_date, cur_step, alphas, lambds, gc_imps, demands, gc_qlt):
        return self.env.allocation(cur_date, cur_step, alphas, lambds, gc_imps, demands, gc_qlt)

    def get_gc_infos(self, cur_date, use_best):
        return self.env.get_gc_infos(cur_date, use_best)

    def run(self):
        cur_episode_cnt = 0
        historical_best_score = 0.

        while True:
            traj = []
            cur_episode_cnt += 1
            self.cur_date = str(np.random.choice(TRAIN_DATE_PERIOD, 1)[0])
            is_evaluation = False
            action_freezing = False
            cur_step = 0

            # episode config
            if IS_EVAL_AND_USE_BEST:  # using best alpha and no action
                is_evaluation = True
            elif IS_TESTING:  # validating model on testing data
                is_evaluation = True
            elif cur_episode_cnt % 10 == 0:  # evaluating current model by best initials
                is_evaluation = True
            elif IS_EVAL_AND_USE_YESTD_BEST:
                is_evaluation = True
            else:  # normal training
                if np.random.random() < ACTION_FREEZING_PROB:
                    action_freezing = True
                    self.env.init_noise.set_sigma(EPSILON)
                else:
                    self.env.init_noise.set_sigma(ALPHA_INIT_BIAS_STDDEV)

            init_step = cur_step

            # result vars
            sum_r_rtb = 0
            sum_r_gc = self.env.date_gc_rgc[self.cur_date]

            # gc bidders vars
            demands, price, lambds, alphas, penalties, opt_outcome = self.get_gc_infos(self.cur_date, use_best=is_evaluation)
            gc_imps = np.repeat([0.], MAX_GC_CNT)
            gc_qlt = np.repeat([0.], MAX_GC_CNT)

            s = get_state(cur_step, demands, penalties, gc_imps, None, None, 0, None)
            a = np.repeat([0.], MAX_GC_CNT).reshape(-1, MAX_GC_CNT)
            dones = np.repeat([False], MAX_GC_CNT).reshape(-1, MAX_GC_CNT)

            while cur_step <= MAX_STEP:
                if cur_step > init_step:
                    if not action_freezing and not IS_EVAL_AND_USE_BEST and not IS_EVAL_AND_USE_YESTD_BEST:
                        a = self.update_alpha(s, alphas, penalties, dones, apply_noise=(not is_evaluation))
                        a = np.array(a).reshape(-1, MAX_GC_CNT)
                        pass

                gc_imps_last = copy.deepcopy(gc_imps)
                gc_qlt_last = copy.deepcopy(gc_qlt)
                allocation_rt = self.allocation(self.cur_date, cur_step, alphas, lambds, gc_imps, demands, gc_qlt)
                if allocation_rt is not None:
                    r_rtb, _, rtb_cpm, gc_pv_ratio = allocation_rt
                else:
                    break

                q_gc = np.sum(gc_qlt - gc_qlt_last)
                ns = get_state(cur_step + 1, demands, penalties, gc_imps, s, rtb_cpm, gc_pv_ratio, gc_imps_last)
                r = np.array([r_rtb + q_gc + sum_r_gc / (MAX_STEP + 1)]).reshape(-1, 1)

                # train model
                if np.random.random() < EPSD_TRAIN_R and (not is_evaluation):
                    loss = self.train()
                    if len(loss) > 0:
                        cur_time = int(datetime.datetime.now().timestamp())
                        log_items = (cur_time, cur_step, loss[0], loss[1], self.is_memory_ready)
                        print("Ts: {}\tstep: {:d}\tq_loss: {:.2f}\ta_loss: {:.2f}\tmemory_ready: {}".format(*log_items))

                # the finished gc bidders will get a -10000 alpha to ensure that no winning impressions any more
                process_finished_gc(alphas, gc_imps, demands)
                dones = np.logical_or((gc_imps >= demands), cur_step == MAX_STEP).reshape(-1, MAX_GC_CNT)
                if cur_step > init_step and (not is_evaluation):
                    traj.append([s, a, r, ns, dones, copy.deepcopy(alphas)])

                sum_r_rtb += r_rtb
                cur_step += 1
                s = ns

            # calculating outcome
            ud_penalties = np.clip(demands - gc_imps, 0, np.iinfo(np.int32).max) * penalties
            sum_penalty = np.sum(ud_penalties)
            gc_qlt_discount = (1 - np.clip(gc_imps - demands, 0, np.iinfo(np.int32).max) / (demands + EPSILON)) * gc_qlt
            sum_q_gc = np.sum(gc_qlt_discount * lambds)
            outcome = sum_r_rtb + sum_q_gc + sum_r_gc - sum_penalty

            # store trajectory
            if traj:
                traj[-1][2] -= sum_penalty
                self.store_traj(traj, outcome, opt_outcome)

            # logging
            cur_time = int(datetime.datetime.now().timestamp())
            rr = outcome / opt_outcome
            log_items = (cur_time, is_evaluation, cur_episode_cnt, rr, outcome, sum_r_rtb, sum_r_gc - sum_penalty, sum_q_gc, sum_penalty, self.is_memory_ready)
            print("Ts: {} Is_eval: {} Episode: {:d} R/R*: {:.2f} R: {:.2f} R_RTB: {:.2f} R_GC: {:.2f} Q_GC: {:.2f} P: {:.2f} memory_ready: {}".format(*log_items))

            # store marl model
            if self.is_memory_ready and is_evaluation and (not IS_TESTING) and outcome / opt_outcome >= historical_best_score:
                historical_best_score = outcome / opt_outcome
                flag = str(round(historical_best_score, 2))
                path = "models/" + str(PUBID) + "_" + TRAIN_DATE_PERIOD[0] + "/" + str(cur_episode_cnt) + "_" + flag + "/"
                os.system("mkdir -p " + path)
                for agent in self.gc_bidders:
                    torch.save(agent.actor, path + str(agent.id))
