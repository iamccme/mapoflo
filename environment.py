# coding: utf8

import pandas as pd
import datetime

from config import *
from noise import *
from itertools import groupby
from collections import defaultdict


class Environment:
    def __init__(self):
        self.date_pv_data = dict()
        self.date_gc_data = dict()
        self.date_gc_info = dict()
        self.date_gc_index = defaultdict(set)
        self.date_gc_index_pos = defaultdict(dict)
        self.date_gc_rgc = defaultdict(int)
        self.date_opt_outcome = dict()
        self.init_noise = NormalActionNoise(mu=0, sigma=ALPHA_INIT_BIAS_STDDEV)
        self.load_data()

    def load_data(self):
        day = str(TRAIN_DATE_PERIOD[0])
        print("Loading:", DATA_PATH + day + ".sample.pv.txt")
        pv_data = pd.read_csv(DATA_PATH + day + ".sample.pv.txt", header=None, delimiter=",|;", engine='python')
        pv_data.sort_values(by=[pv_data.shape[1] - 1], inplace=True)
        pv_data = pv_data.to_numpy()
        self.date_pv_data[day] = pv_data[pv_data[:, 0] == PUBID, :]
        self.date_pv_data[day] = [np.array(list(j)) for i, j in groupby(self.date_pv_data[day], lambda p: p[-1]) if i]
        gc_index = dict()
        for i, j in enumerate(self.date_pv_data[day][0][0, 1:1 + MAX_GC_CNT]):
            gc_index[j] = i

        gc_date = day
        if not (IS_TESTING or IS_EVAL_AND_USE_YESTD_BEST):
            gc_date = (datetime.datetime.strptime(day, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

        print("Loading:", DATA_PATH + gc_date + ".gc.txt")
        gc_data = pd.read_csv(DATA_PATH + gc_date + ".gc.txt", header=None)
        gc_data = gc_data.to_numpy()
        self.date_gc_data[day] = gc_data[gc_data[:, 0] == PUBID, :]
        self.date_gc_info[day] = dict()
        for row in self.date_gc_data[day]:
            # row format: pub_id,adv_id,demand,price,penalty,lambda,optimal_alpha,opt_obj
            row[2] /= 10  # due to the 10% sampling
            gc_id = row[1]
            self.date_gc_index[day].add(gc_index[gc_id])
            self.date_gc_info[day][gc_id] = row[2:]
            self.date_gc_rgc[day] += row[2] * row[3]
            self.date_opt_outcome[day] = row[7]

        pos = 0
        for i in sorted(list(self.date_gc_index[day])):
            self.date_gc_index_pos[day][i] = pos
            pos += 1

    def allocation(self, cur_date, cur_step, alphas, lambds, gc_imps, demands, gc_qlt):
        if cur_step >= len(self.date_pv_data[cur_date]):
            print("[ERROR] cur_step not in PV data! ", cur_step)
            return None

        pv_data = self.date_pv_data[cur_date][cur_step]
        rtb_bids = pv_data[:, MAX_GC_CNT + MAX_GC_CNT + 1]
        gc_bids = alphas + pv_data[:, MAX_GC_CNT + 1:MAX_GC_CNT + MAX_GC_CNT + 1] * lambds

        return self.allocation_with_bid(gc_bids, pv_data, rtb_bids, lambds, gc_imps, demands, gc_qlt)

    def allocation_with_bid(self, gc_bids, pv_data, rtb_bid, lambds, gc_imps, demands, gc_qlt):
        max_gc_bids = np.max(gc_bids, axis=1) * 1000
        max_gc_bids_index = np.argmax(gc_bids, axis=1)
        max_gc_bids_pctr = pv_data[:, MAX_GC_CNT + 1:MAX_GC_CNT + MAX_GC_CNT + 1][range(len(pv_data)), max_gc_bids_index]

        is_gc_win = max_gc_bids > rtb_bid
        is_gc_lose = (1 - is_gc_win).astype(np.bool)

        r_rtb_list = pv_data[is_gc_lose, MAX_GC_CNT + MAX_GC_CNT + 1]
        max_gc_bids_index_gc_win = max_gc_bids_index[is_gc_win]
        q_gc_list = (pv_data[:, 1 + MAX_GC_CNT:1 + MAX_GC_CNT + MAX_GC_CNT] * lambds)[is_gc_win, max_gc_bids_index_gc_win]

        r_rtb = np.sum(r_rtb_list) / 1000
        q_gc = np.sum(q_gc_list)

        rtb_cpm = float(r_rtb) / len(r_rtb_list) * 1000 if len(r_rtb_list) > 0 else 0.0
        gc_pv_ratio = np.sum(is_gc_win) / float(len(pv_data))

        # get bidder with max bid and winning status
        t = np.array([max_gc_bids_index, is_gc_win.astype(np.int), max_gc_bids_pctr * is_gc_win.astype(np.int)]).transpose()
        # sort by bidder
        t = t[np.lexsort((t[:, 0],)), :]
        # get bidder index with its win cnt
        gc_win_cnt = np.array([(i, np.sum(np.array(list(j))[:, 1])) for i, j in groupby(t, lambda p: p[0])])
        gc_win_pctr = np.array([(i, np.sum(np.array(list(j))[:, 2])) for i, j in groupby(t, lambda p: p[0])])
        # update gc win cnt
        gc_imps[gc_win_cnt[:, 0].astype(np.int32)] += gc_win_cnt[:, 1]
        # update gc win qualities
        gc_qlt[gc_win_cnt[:, 0].astype(np.int32)] += gc_win_pctr[:, 1]

        return r_rtb, q_gc, rtb_cpm, gc_pv_ratio

    def get_gc_infos(self, cur_date, use_best):
        demands = np.repeat([0.], MAX_GC_CNT)
        price = np.repeat([0.], MAX_GC_CNT)
        lambds = np.repeat([0.], MAX_GC_CNT)
        alphas = np.repeat([-10000.], MAX_GC_CNT)
        penalties = np.repeat([0.], MAX_GC_CNT)
        opt_outcome = 0
        bidders = self.date_pv_data[cur_date][0][0, 1:MAX_GC_CNT + 1]

        for i in range(len(bidders)):
            # date_gc_info format: demand,price,penalty,lambda,optimal_alpha,opt_obj,gc_pv,gc_click,rtb_pv,rtb_revnue,gc_revnue,gc_qulality,gc_penalty
            if bidders[i] in self.date_gc_info[cur_date]:
                demands[i] = self.date_gc_info[cur_date][bidders[i]][0]
                price[i] = self.date_gc_info[cur_date][bidders[i]][1]
                lambds[i] = self.date_gc_info[cur_date][bidders[i]][3]
                penalties[i] = self.date_gc_info[cur_date][bidders[i]][2]
                if use_best:
                    alphas[i] = self.date_gc_info[cur_date][bidders[i]][4]
                else:
                    alphas[i] = self.date_gc_info[cur_date][bidders[i]][4] + (penalties[i] * self.init_noise())
                opt_outcome = self.date_opt_outcome[cur_date]

        return demands, price, lambds, alphas, penalties, opt_outcome


if __name__ == '__main__':
    pass
