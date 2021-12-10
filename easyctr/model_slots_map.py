#! /usr/bin/env python
# -*-coding=utf8-*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


MODEL_SLOTS_MAP = {
    'wide': ['linear'],
    'lr': ['linear'],
    'linear': ['linear'],
    'dnn': ['dnn'],
    'deep': ['dnn'],
    'wide_deep': ['dnn', 'linear'],
    'wdl': ['dnn', 'linear'],
    'multi_dnn': ['multi_dnn'],
    'multi_deep': ['multi_dnn'],
    'multi_wide_deep': ['multi_dnn', 'linear'],
    'multi_wdl': ['multi_dnn', 'linear'],
    'fm': ['fm', 'linear'],
    'fwfm': ['fwfm', 'linear'],
    'afm': ['afm', 'linear'],
    'iafm': ['iafm', 'linear'],
    'ifm': ['ifm', 'linear'],
    'kfm': ['kfm', 'linear'],
    'wkfm': ['wkfm', 'linear'],
    'nifm': ['nifm', 'linear'],
    'cross': ['cross', 'linear'],
    'cin': ['cin', 'linear'],
    'autoint': ['autoint', 'linear'],
    'autoint+': ['autoint', 'dnn', 'linear'],
    'nfm': ['nfm', 'linear'],
    'nkfm': ['nkfm', 'linear'],
    'ccpm': ['ccpm', 'linear'],
    'ipnn': ['ipnn', 'linear'],
    'kpnn': ['kpnn', 'linear'],
    'pin': ['pin', 'linear'],
    'fibinet': ['fibinet', 'linear'],
    'deepfm': ['dnn', 'fm', 'linear'],
    'deepfwfm': ['dnn', 'fwfm', 'linear'],
    'deepafm': ['dnn', 'afm', 'linear'],
    'deepiafm': ['dnn', 'iafm', 'linear'],
    'deepifm': ['dnn', 'ifm', 'linear'],
    'deepkfm': ['dnn', 'kfm', 'linear'],
    'deepwkfm': ['dnn', 'wkfm', 'linear'],
    'deepnifm': ['dnn', 'nifm', 'linear'],
    'xdeepfm': ['dnn', 'cin', 'linear'],
    'dcn': ['dnn', 'cross', 'linear'],
    'deepcross': ['dnn', 'cross', 'linear'],
}
