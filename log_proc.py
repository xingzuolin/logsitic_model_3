#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 15:54
# @Author  : Jun
# @File    : log_proc.py


import os
import sys


def curr_file_num():
    curr_dir_file = sys.argv[0]
    try:
        curr_file_name = curr_dir_file[curr_dir_file.rindex("/") + 1:curr_dir_file.rindex(".")]
    except:
        curr_file_name = curr_dir_file[:curr_dir_file.rindex(".")]
    return curr_file_name


def clear_log():
    curr_script_name = curr_file_num()
    for ext in ('.log', '.lst'):
        if os.path.isfile(curr_script_name + ext):
            os.remove(curr_script_name + ext)


def append_logs():
    curr_script_name = curr_file_num()
    return [open(curr_script_name + '.log', 'a'), open(curr_script_name + '.lst', 'a')]