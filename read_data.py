#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 15:52
# @Author  : Jun
# @File    : read_data.py


import re
import numpy as np
import pandas as pd
import time
from log_proc import *


def read_hive_data(infile, top_n_rows = None, data_label = None, out_hdf5 = None):

    log_file, lst_file = append_logs()

    if re.search('.gz$', infile):
        zip_tag = 'gzip'
    elif re.search('.bz2$', infile):
        zip_tag = 'bz2'
    else:
        zip_tag = None

    head_type = pd.read_table(infile, sep='\t', compression=zip_tag, header=None, skiprows=None, nrows=1)
    head_type_t = head_type.T.fillna('')

    var_name_lst = []
    var_type_lst = []

    for idx in range(len(head_type_t)):
        name_type_str = head_type_t.ix[idx, 0]
        if name_type_str != '':
            var_name, var_type = name_type_str.split('|')
            var_name_lst.append(var_name.upper())
            if var_type.lower() == 'string':
                var_name_lst.append(np.string_)
            elif var_type.lower() == 'number':
                var_type_lst.append(np.float64)
            else:
                var_type_lst.append(np.string_)

    start_time = time.time()

    df = pd.read_table(infile, sep='\t', compression=zip_tag, header=None, skiprows=1, nrows=top_n_rows,
                       names=var_name_lst, dtype=dict(zip(var_name_lst, var_type_lst)))

    log_file.write("{0} records were read from the infile {1}".format(len(df.index), infile) + '\n')
    log_file.write("The returned DataFrame has {0} observations and {1} columns".format(len(df.index), len(df.columns)) + '\n')
    log_file.write("Time Cost: {:.2f} seconds".format(time.time() - start_time) + '\n')

    if data_label is not None:
        label_df = pd.read_table(data_label, sep=',', header=None, skiprows=1, names=['variable', 'label'])

    try:
        content_file = infile[infile.rindex('/') + 1:]
    except:
        content_file = infile

    f_content = open(content_file + '.content.txt', 'w')

    for element in df.columns:
        if df[element].dtype == object:
            df[element] = df[element].fillna('_null_')
        if data_label is not None:
            label = list(label_df[label_df['variable'] == element]['label'].values)[0]
            f_content.write("%-35s\t%-20s\t%-20s\n" % (element, df[element].dtype, label))
        else:
            f_content.write("%-35s\t%-20s\n" % (element, df[element].dtype))

    log_file.close()
    lst_file.close()
    f_content.close()

    if out_hdf5 is not None:
        h5_file = pd.HDFStore(out_hdf5, 'w', complevel=9, complib='zlib')
        h5_file['data'] = df
        if data_label is not None:
            h5_file['label'] = label_df
        h5_file.close()
    return df



