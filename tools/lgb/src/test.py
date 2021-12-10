# -*-coding=utf8-*-
import feature as ft
import columns as cl
import lightgbm as lgb
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
ft.get_dict_map()
for i in ft.dict_map:
    print(i)
    print("\n")
