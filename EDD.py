#!/usr/bin/env python
# @Author  : Jun
# @Time    : 2018/3/10 11:46
# @File    : EDD.py
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import time
import sys
from log_proc import *
from func import *
import math

def num_edd(var_series):
    stats_dict = {}
    stats_dict['n_valid'] = var_series.count()
    stats_dict['n_miss'] = len(var_series) - var_series.count()
    stats_dict['unique'] = len(var_series.unique())
    stats_dict['n_tot'] = len(var_series)
    stats_dict['miss_pct'] = round(stats_dict['n_miss']/float(len(var_series)), 4)

    if len(var_series.unique()) == 1 and str(var_series.unique()[0]) == 'nan':
        stats_dict['mean_or_top1'] = '.'
        stats_dict['min_or_top2'] = '.'
        stats_dict['p1_or_top3'] = '.'
        stats_dict['p5_or_top4'] = '.'
        stats_dict['p25_or_top5'] = '.'
        stats_dict['median_or_bot5'] = '.'
        stats_dict['p75_or_bot4'] = '.'
        stats_dict['p95_or_bot3'] = '.'
        stats_dict['p99_or_bot2'] = '.'
        stats_dict['max_or_bot1'] = '.'

    else:
        stats_dict['mean_or_top1'] = var_series.mean()
        stats_dict['min_or_top2'] = var_series.min()
        stats_dict['p1_or_top3'] = var_series.quantile(0.01)
        stats_dict['p5_or_top4'] = var_series.quantile(0.05)
        stats_dict['p25_or_top5'] = var_series.quantile(0.25)
        stats_dict['median_or_bot5'] = var_series.quantile(0.5)
        stats_dict['p75_or_bot4'] = var_series.quantile(0.75)
        stats_dict['p95_or_bot3'] = var_series.quantile(0.95)
        stats_dict['p99_or_bot2'] = var_series.quantile(0.99)
        stats_dict['max_or_bot1'] = var_series.max()

    return stats_dict


def char_edd(var_series):
    stats_dict = {}
    freq_stats_df = DataFrame(var_series.value_counts())
    freq_stats_df.columns = ['count']
    freq_stats_df.sort_values(by='count', ascending=False, inplace=True)
    try:
        stats_dict['n_valid'] = freq_stats_df['count'].sum() - freq_stats_df.ix['_null_', 'count']
        stats_dict['n_miss'] = freq_stats_df.ix['_null_', 'count']
    except:
        stats_dict['n_valid'] = freq_stats_df['count'].sum()
        stats_dict['n_miss'] = 0
    stats_dict['n_tot'] = len(var_series)
    stats_dict['miss_pct'] = round(stats_dict['n_miss'] / float(len(var_series)), 4)
    stats_dict['unique'] = len(freq_stats_df.index.unique())

    char_stat_lst = ['mean_or_top1', 'min_or_top2', 'p1_or_top3', 'p5_or_top4', 'p25_or_top5', 'median_or_bot5',
                     'p75_or_bot4', 'p95_or_bot3', 'p99_or_bot2', 'max_or_bot1']

    top_cat_stats = freq_stats_df[:5]
    bottom_cat_stats = freq_stats_df[-5:]

    for seq in range(len(top_cat_stats)):
        stats_dict[char_stat_lst[seq]] = freq_stats_df.ix[seq].name + '::' + str(freq_stats_df.ix[seq, 'count'])

    for seq in range(len(bottom_cat_stats)):
        seq -= len(bottom_cat_stats)
        stats_dict[char_stat_lst[seq]] = freq_stats_df.ix[seq].name + '::' + str(freq_stats_df.ix[seq, 'count'])

    return stats_dict


def edd(data_df, output, include_var_list=None, exclude_var_list=None):
    log_file, lst_file = append_logs()
    for x in log_file, lst_file:
        x.write('\n')

    start_time = time.time()
    var_name_list = [col.upper() for col in data_df.columns]

    if include_var_list is None:
        include_var_set = set(var_name_list)
    else:
        include_var_list = [var.upper() for var in include_var_list ]
        overlap_set = set(var_name_list).intersection(include_var_list)
        if len(overlap_set) == 0:
            log_file.write('None of the variables specified in include_var_list is in given data frame! \n')
            exit(1)
        else:
            include_var_set = overlap_set
    if exclude_var_list is None:
        exclude_var_list = []
    exclude_var_set = set([var.upper() for var in exclude_var_list])
    include_var_set -= exclude_var_set

    var_name_lst = list(include_var_set)
    var_format_lst = ['string' if str(data_df[x].dtype).find('object') >= 0 else str(data_df[x].dtype) for x in
                      var_name_lst]
    var_type_lst = ['CHAR' if x == 'string' else 'NUM' for x in var_format_lst]
    var_info_base_df = DataFrame(dict(zip(['name', 'type', 'format'], [var_name_lst, var_type_lst, var_format_lst])),
                                 index=var_name_lst, columns=['name', 'type', 'format'])
    var_info_base_df['position'] = range(len(var_name_lst))
    edd_dict = {}
    for var in var_info_base_df.index:
        if var_info_base_df.ix[var, 'type'] == 'NUM':
            log_file.write('Processing numerical {0}...\n'.format(var))
            edd_dict[var] = num_edd(var_series=data_df[var])
            log_file.write('Numerical {0} finished.\n'.format(var))
        else:
            log_file.write('Processing character {0}..\n.'.format(var))
            edd_dict[var] = char_edd(var_series=data_df[var])
            log_file.write('Character {0} finished.\n'.format(var))

    edd_df = DataFrame(edd_dict).T
    final_edd_df = var_info_base_df.merge(edd_df, left_on='name', right_index=True, how='left')
    final_edd_df.to_csv(output, index=False)
    log_file.write("Time Cost: {:.2f} seconds".format(time.time() - start_time) + '\n')
    log_file.close()
    lst_file.close()
    return final_edd_df


def keep_columns(data_df, var_stats, miss_pct=0.85):
    drop_var_stats = list(var_stats.ix[var_stats['miss_pct'] >= 0.85, 'name'])
    data_df_col_names = data_df.columns
    keep_vars = set(data_df_col_names) - set(drop_var_stats)
    return data_df.ix[:, keep_vars]


def _bin_numbers(col1, col2, bin_n):
    """
    Creates bin_n bins based on both col1 and col2
    """
    col1 = col1[~col1.map(lambda x: is_null_flag(x))].reset_index(drop=True)
    col2 = col2[~col2.map(lambda x: is_null_flag(x))].reset_index(drop=True)
    comb = pd.Series(pd.np.concatenate([col1, col2])).sort_values(inplace=False).reset_index(drop=True)
    bin_size = int(len(comb) / bin_n)
    bin_dict1, bin_dict2 = {}, {}
    for i in range(bin_n - 1):  # last bin only needs bin_min
        bin_low = comb[i*bin_size]
        bin_high = comb[(i+1)*bin_size]
        bin_dict1[i] = sum((col1 >= bin_low) & (col1 < bin_high))
        bin_dict2[i] = sum((col2 >= bin_low) & (col2 < bin_high))
        # print bin_low, bin_high
    # Highest bin
    bin_dict1[i+1] = sum(col1 >= bin_high)
    bin_dict2[i+1] = sum(col2 >= bin_high)
    return bin_dict1, bin_dict2


def _bin_char(col1, col2, bin_n):
    """
    Creates bin_n bins based on both col1 and col2
    """
    col1 = col1[~col1.map(lambda x: is_null_flag(x))].reset_index(drop=True)
    col2 = col2[~col2.map(lambda x: is_null_flag(x))].reset_index(drop=True)
    comb = pd.Series(pd.np.concatenate([col1, col2])).sort_values(inplace=False).reset_index(drop=True)
    comb_var = list(comb.value_counts().index)
    bin_dict1, bin_dict2 = {}, {}
    for i in range(len(comb_var)):
        bin_dict1[i] = col1[col1 == i].sum()
        bin_dict2[i] = col2[col1 == i].sum()
    return bin_dict1, bin_dict2


def _is_number_list(alist):
    def _is_number(obj):
        try:
            obj + 0
            return True
        except TypeError:
            return False
    for obj in alist:
        if not _is_number(obj):
            return False
    return True


def _distr_stat(col1, col2, f):
    """
    Calculate a distribution based stat based on bins
    """
    bin_threshold = 10
    vcs1, col1_len = col1.value_counts().to_dict(), float(len(col1))
    vcs1["_Empty_"] = sum(col1.map(lambda x: is_null_flag(x)))
    vcs2, col2_len = col2.value_counts().to_dict(), float(len(col2))
    vcs2["_Empty_"] = sum(col2.map(lambda x: is_null_flag(x)))
    values = set.union(set(vcs1.keys()), set(vcs2.keys()))
    stat = 0
    if len(values) <= bin_threshold:
        for v in values:
            v_share1 = (vcs1.get(v, 0)+1)/col1_len
            v_share2 = (vcs2.get(v, 0)+1)/col2_len
            stat += f(v_share1, v_share2)
    else:
        null_share1 = (vcs1.pop("_Empty_")+1)/col1_len
        null_share2 = (vcs2.pop("_Empty_")+1)/col2_len
        if null_share1 >= 1 or null_share2 >= 1:
            stat += f(null_share1, null_share2)
        elif _is_number_list(vcs1.keys()) and _is_number_list(vcs2.keys()):
            bins1, bins2 = _bin_numbers(col1, col2, bin_threshold)
            for v in range(bin_threshold):
                v_share1 = (bins1.get(v, 0)+1)/col1_len
                v_share2 = (bins2.get(v, 0)+1)/col2_len
                stat += f(v_share1, v_share2)
            stat += f(null_share1, null_share2)
        else:
            bins1, bins2 = _bin_char(col1, col2, bin_threshold)
            bin_char_threshold = max(len(bins1.keys()), len(bins2.keys()))
            for v in range(bin_char_threshold):
                v_share1 = (bins1.get(v, 0)+1)/col1_len
                v_share2 = (bins2.get(v, 0)+1)/col2_len
                stat += f(v_share1, v_share2)
            stat += f(null_share1, null_share2)
        # else:
        #     stat = "Too many unique values or not numbers."  # excepted by TypeError (round)
    print(stat)
    return round(stat, 4)


def calc_pair_stat(col1, col2, stat):
    if stat == "psi":
        return _distr_stat(col1, col2, _psi_function)


def _psi_function(share1, share2):
    """
    PSI = SIGMA_i ((A_i - B_i)*ln(A_i/B_i)) where A_i/B_i is share of i-th bin
    """
    return (share1 - share2) * math.log(share1/share2)


def cal_psi(data_m, data_o, output, ignore_null=False, exclude_var=[], stat='psi'):
    data_m_col = data_m.columns
    data_o_col = data_o.columns
    com_col_set = set(data_m_col).intersection(set(data_o_col))
    if len(exclude_var) > 0:
        exclude_var_set = set([var.upper() for var in exclude_var])
        com_col_set = com_col_set - exclude_var_set
    common_cols = list(com_col_set)
    if len(common_cols) == 0:
        print('there is no common vars from the model and oot')
        exit(1)
    stat_lst = []
    stat_dict = {}
    for col in common_cols:
        print(col)
        # try:
        if ignore_null:
            pair_stat = calc_pair_stat(data_m[col].dropna(), data_o[col].dropna(), stat)
        else:
            pair_stat = calc_pair_stat(data_m[col], data_o[col], stat)
        stat_lst.append([col, pair_stat])
        stat_dict[col] = pair_stat
        # except TypeError:
        #     stat_lst.append([col, None])
        #     stat_dict[col] = None
    df_ = pd.DataFrame(stat_lst, columns=['variable', 'psi'])
    df_.sort_values(by='psi', ascending=False, inplace=True)
    df_.to_csv(output, index=False)
    return df_
