#! /usr/bin/env python
# -*- coding=utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


cmd = 'bash ./test.sh'
output = os.popen(cmd)
print('\n'.join(output.readlines()))
