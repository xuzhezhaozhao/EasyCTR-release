#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import requests
import sys


"""上报到企业微信"""


if len(sys.argv) != 4:
    print("Usgae: <receiver> <title> <content_file>")
    sys.exit(-1)


receiver = sys.argv[1]
title = sys.argv[2]
text = open(sys.argv[3]).read().strip().replace('\n', '\\n')

data = '{"receiver":"%s","msg":"%s", "title":"%s"}' % (receiver, text, title)
requests.post(url='http://t.isd.com/api/sendQiYeWX', data=data)
