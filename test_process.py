#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 15:59
# @Author  : Jun
# @File    : test.py

import os
import sys
import numpy as np
import pandas as pd
from log_proc import *
from read_data import read_hive_data
import func


# src = r'E:/dev_code/logistic_data/data/'
#
# tgt = r'E:/dev_code/logistic_data/modeling/data/'
#
# clear_log()
#
# collection_mst = read_hive_data(infile=src+'collection_scr_20141120.gz', top_n_rows=None, data_label=src+'data_label_qf.csv', out_hdf5=tgt+'collection.h5')


print('-------------------------step01 woe--------------------------------')

clear_log()
src = r'E:/dev_code/logistic_data/modeling/data/'
seg = 1

unit_weight = 'UNIT_WEIGHT'
doll_weight = 'DOLL_WEIGHT'
target_var = 'IS_COLLECTION_BAD'

prefix = 'collection'

if not os.path.isdir(src+'s'+str(seg)):
    os.mkdir(src+'s'+str(seg))
if not os.path.isdir(src+'s'+str(seg)+'/woe_files'):
    os.mkdir(src+'s'+str(seg)+'/woe_files')

tgt = src+'s'+str(seg)
h5 = pd.HDFStore(src+'collection.h5', 'r')
collection_mst = h5['data']
data_label = h5['label']
h5.close()

dev, oot = func.dev_oot_split(collection_mst, seg_no=seg, target_var=target_var)

print(dev[target_var].value_counts())
print(oot[target_var].value_counts())

h5 = pd.HDFStore(tgt + '/s' + str(seg) + '_dev_oot.h5', 'w')
h5['dev'] = dev
h5['oot'] = oot
h5.close()

