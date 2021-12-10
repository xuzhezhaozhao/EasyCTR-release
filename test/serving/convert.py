import os
import sys
import tensorflow as tf
from tensorflow.core.framework.step_stats_pb2 import StepStats
from tensorflow.python.client import timeline

"""
Convert C++ profile log to timeline format.
"""

filename = 'profile.log'

step_stats = StepStats()
with open(filename, 'rb') as f:
    step_stats.ParseFromString(f.read())

x = timeline.Timeline(step_stats).generate_chrome_trace_format()
with open('profile.timeline', 'w') as f :
    f.write(x)
