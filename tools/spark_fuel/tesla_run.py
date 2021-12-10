#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess


def call(cmd):
    print("execute cmd '{}' ...".format(cmd))
    x = subprocess.call(cmd, shell=True)
    return x


call('bash ./tesla_run.sh')
