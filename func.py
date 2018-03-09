#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 17:47
# @Author  : Jun
# @File    : func.py


import pandas as pd
import numpy as np


def dev_oot_split(mst_df, seg_no, target_var):
    target_unique = sorted(list(mst_df[target_var].unique()))
    dev = pd.DataFrame()
    oot = pd.DataFrame()
    for val in target_unique:
        mst_seg = mst_df[(mst_df['SEG'] == seg_no) & (mst_df[target_var] == val)]
        random_idx_seq = np.random.permutation(len(mst_seg))
        dev_tmp = mst_seg.take(random_idx_seq)[:int(0.51 * len(random_idx_seq))]
        oot_tmp = mst_seg.take(random_idx_seq)[int(0.51 * len(random_idx_seq)):]
        dev = pd.concat([dev, dev_tmp], axis=0)
        oot = pd.concat([oot, oot_tmp], axis=0)
    return dev, oot


